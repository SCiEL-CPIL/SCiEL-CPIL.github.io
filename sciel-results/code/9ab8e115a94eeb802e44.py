import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0

def acquisition(mu, sigma, f_best, cost):
    if (time.time() - _START_TIME) > _DEADLINE_SECONDS:
        return -np.asarray(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # UCB for minimization: -mu + beta * sigma
    # We choose beta adaptively. 
    # A common choice is beta = sqrt(2 * log(t)). 
    # Since we don't have t, we can use a heuristic based on the data.
    # Or a fixed value. 
    # For Rastrigin, exploration is important.
    # Let's try beta = 1.5.
    
    beta = 1.5
    ucb = -mu + beta * sigma
    
    scores = ucb / cost
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
