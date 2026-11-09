import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 280.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return -mu
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge case where f_best is very small
    # If f_best is 0, we are done, but let's be safe.
    if f_best <= 1e-9:
        # Pure exploitation
        return -mu
    
    # Quantile normalization of sigma
    # This maps sigma values to [0, 1] based on their rank
    # This is robust to the scale of sigma
    n = len(sigma)
    # np.argsort(np.argsort(x)) gives the rank of each element
    # Ties are handled by the order in argsort, which is fine for a heuristic
    ranks = np.argsort(np.argsort(sigma))
    sigma_norm = ranks / (n - 1)
    
    # Dynamic exploration weight
    # Scale exploration with the current best value
    # As f_best approaches 0, exploration decreases
    # We use a base scale of 1.0
    beta = 1.0 * f_best
    
    # Thompson Sampling / UCB Heuristic
    # Score = -mu + beta * sigma_norm
    # Higher score is better
    scores = -mu + beta * sigma_norm
    
    # Ensure finite scores
    # If any NaNs or Infs appear, replace with a very low score
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
