import numpy as np

def acquisition(mu, sigma, f_best):
    # Ensure mu is a numpy array
    mu = np.asarray(mu)
    
    # We want to minimize the function, so we want to maximize the score.
    # The strategy is to pick the candidate with the lowest posterior mean.
    # Score = -mu
    
    # Handle non-finite values
    scores = -mu
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
