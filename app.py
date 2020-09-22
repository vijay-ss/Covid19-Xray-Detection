# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

import numpy as np
import h5py
from PIL import Image
import PIL
import os

# Initiate Flask app
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'

# Prep Keras model
MODEL_ARCHITECTURE = './covid19_model_adv.json'
MODEL_WEIGHTS = './covid19_model_weights.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
print ('Keras model loaded.')

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print ('Keras model weights loaded.')

# take input image and make prediction
def predict(input_image):
    img = image.load_img(input_image, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict_classes(x)
    return prediction

# Flask app routes
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == "POST":

        # Save file from user upload
        # image_file = request.files["file"]
        # if image_file:
        #     image_location = os.path.join(
        #         UPLOAD_FOLDER,
        #         image_file.filename
        #     )
        #     image_file.save(image_location)
        #     print(image_file, image_location)


        # Get the file from post request
        f = request.files['file']

		# Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
       
        pred = predict(file_path)
        print(pred)

        # Define the class type
        classes = {'Diagnosis': ['Covid-19', 'Healthy']}

        # Return the class type based on prediction
        predicted_class = classes['Diagnosis'][pred[0][0]]
        print('Prediction: {}.'.format(predicted_class))
        return str(predicted_class)

if __name__ == '__main__':
	app.run(debug = True)