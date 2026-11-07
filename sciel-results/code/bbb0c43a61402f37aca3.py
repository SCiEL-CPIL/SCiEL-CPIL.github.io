import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling for minimization
    # We sample a value from the posterior distribution for each candidate
    # and return the negative of that sample as the score.
    # The evaluator picks the candidate with the highest score,
    # which corresponds to the candidate with the lowest sampled objective value.
    
    # Ensure sigma is positive to avoid issues with sqrt if any numerical noise made it negative
    # Though GP sigma should be positive by construction.
    sigma = np.maximum(sigma, 1e-8)
    
    # Sample from N(mu, sigma^2)
    samples = np.random.normal(mu, sigma)
    
    # We want to minimize, so we maximize the negative sample
    score = -samples
    
    return score
