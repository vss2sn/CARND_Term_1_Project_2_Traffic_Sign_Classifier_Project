# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 86199 (Augmented using balance classes) & 34799 (without augmentation)
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43 

#### 2. Include an exploratory visualization of the dataset.

Visualisation of one image from each of the classes is provided (random selection of an image within a class)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale, as the colour of the images provide little information that aids in distinguishing. They are normalised to 0-1. I did want to try using hitogram equalisation to make sure the contrast in the images was both good and consistent, but it seems a superlative step for the training examples provided, givent he ir accuracy on the test data, and it also consumes quitre a lot of resources, given the size of the data, as it is a little difficult to vectorise the operation. I however, will be of use when driving in bad lighting conditions, and I will develop this line of thought when I have a bit more time.

The main step I took was to balance the classes. This increased the sample size to twice it's orriginal size, but prevents biased training, as if there are very few examples of a class, the network does not learn to recognise them easily, leading to incorrect classification of those images. The new data generated was using images from the same class, but with it's persective changed using opencv. This allows the netwrok access to an image it has not seen before and maeks the identification more resilient different view points of the same sign.

To sum up, the difference between the original data set and the augmented data set is:
1. Larger training set
2. New images are based on, but not the same as initial data set
3. All the pixel values in all the images range from 0 to 1 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Preprocessing			| 32x32 Grayscale image							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling 			| 2x2 stride, outputs 5x5x32 				    |
| Fully connected		| flatten, 512 -> 256    						|
| Droupout				| prob = 0.8		    						|
| Fully connected		| 256 -> 43			    						|
|						|												|
|:---------------------:|:---------------------------------------------:|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:
# network parameters
mu = 0
sigma = 0.1

# training parameters
epochs = 20
batch_size = 128

# hyperparameters
learning_rate = 0.001

# optimiser
Adam optimiser

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of peaks at 0.959 with an average over 0.94
* test set accuracy of 0.946

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
Modified LeNet
* Why did you believe it would be relevant to the traffic sign application?
1. http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
2. Good starting point
3. Changed depth of filters and number of neurons in fully connected layers
4. Modified type of initialisation based on reading
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
1. Accuracy increases initially in the validations set and then tapers off. 
2. Proven overfitting avoided to some extent by using warped persective balanced classes, which randomly provide either the left or right persective, but accuracy is consistnec across runs. 
3. Grayscale also makes network resilient to changes in lighting  conditions

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

#(REF: https://github.com/CYHSM/carnd/tree/master/CarND-Traffic-Sign-Classifier-Project/extra_images)
![alt text][/extra_images/1.jpg] ![alt text][/extra_images/2.jpg] ![alt text][/extra_images/3.jpg] 
![alt text][/extra_images/4.jpg] ![alt text][/extra_images/5.jpg]

1. Never encountered some signs before
2. image quality, especially with lot of noisy background

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| Animal	   			| Double curve									|
| No entry      		| No entry 										|
| No passing 			| No passing 									|
| 30 km/h zone     		| Roundabout Mandatory							|
|:---------------------:|:---------------------------------------------:|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The test set accuracy is significantly better.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook, which also generates the graphs for probability comparison.
1. Maybe colour could have helped in this one, as difference in this class and next is a probability of 0.1, not as much of a diff as would be preferred
2. Never seen before sign (expected probabilty to e spread across all possible classes)
3. Correct identification
4. Correct identification
5. Incorrect identification, seems to focus on cirle and returns roundabout mandatory, colour might help

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


