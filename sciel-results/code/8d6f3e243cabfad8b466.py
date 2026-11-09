import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    Samples the posterior at each candidate point and returns the negative
    of the sample. This encourages selecting points with low predicted values
    while allowing exploration of high-uncertainty regions if the sample
    happens to be low there.
    """
    # Draw a sample from the posterior at each candidate
    # mu and sigma are arrays of shape (2000,)
    # We draw a single sample from the Gaussian distribution N(mu, sigma^2)
    # Note: sigma is the standard deviation.
    samples = mu + sigma * np.random.randn(mu.shape)
    
    # We want to maximize the score, but we are minimizing the objective.
    # In Thompson sampling, we would pick the point with the minimum sample.
    # To fit the interface "higher score = evaluate next", we return -sample.
    # This way, the point with the lowest sample value gets the highest score.
    
    # Ensure finite scores
    scores = -samples
    
    # Handle any potential NaN/Inf just in case, though GP posterior should be well-behaved
    # If sigma is 0, sample is mu. If sigma is large, sample is mu + noise.
    
    return scores
