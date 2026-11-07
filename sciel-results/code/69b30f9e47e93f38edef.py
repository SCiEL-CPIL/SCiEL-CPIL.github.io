import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # We want to minimize the objective function.
    # We sample from the posterior distribution for each candidate.
    # The candidate with the lowest sample is the most promising.
    # The acquisition score should be high for the best candidate.
    # Therefore, we return the negative of the samples.
    
    # Ensure sigma is positive to avoid division by zero or negative variance issues in random sampling
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw standard normal random numbers
    z = np.random.randn(len(mu))
    
    # Compute samples from the posterior: y ~ N(mu, sigma^2)
    samples = mu + sigma_safe * z
    
    # Return negative samples so that higher score corresponds to lower predicted value
    return -samples
