# Siamese Network
## A model that compares two images of famous landmarks to determine if they are of the same place
### (If Github doesn't render the Jupyter notebook, it can be viewed here: https://nbviewer.org/github/tzviblonder/Siamese-Network/blob/main/Siamese%20Network.ipynb)

#### A siamese neural network is one that encodes inputs in output vectors. The name comes from the way that the same model is used on two different inputs in order to compare their output.
#### This one is trained using triplet loss, which finds the Euclidean distance between the encoding of the input image and that of one from the same class and compares it to the distance between the input's encoding and that of one from a different class. The goal is the minimize the former and maximize the latter. After the model was trained on a GPU, the training images were used to determine a value that serves as a cuttoff point between images predicted to be of the same place and images predicted to be different. When tested, the model was shown to have 99% accuracy.

#### Models such as this are useful in classifying images, and are common in facial recognition systems.
