import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling strategy
    # We draw a single sample from the posterior distribution N(mu, sigma^2) for each candidate.
    # The candidate with the lowest sampled value is expected to have the lowest true value.
    # Since we want "higher score = evaluate next", we return the negative of the sample.
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw one sample per candidate
    # np.random.normal(mu, sigma_safe) generates an array of shape (2000,)
    # This introduces stochasticity that encourages exploring high-variance regions
    # when the mean is not decisively lower, and exploiting low-variance low-mean regions otherwise.
    
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # Return negative samples so that the minimum value (best) gets the highest score
    return -samples
