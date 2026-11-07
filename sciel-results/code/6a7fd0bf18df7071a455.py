import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    Returns scores where higher is better (i.e., higher expected lower bound).
    score = -mu + beta * sigma
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT_LIMIT - _SAFETY_MARGIN:
        # Fallback to greedy exploitation if running out of time
        # Return improvement over f_best, which is high if mu is low
        return f_best - mu

    # Ensure sigma is non-negative and finite to avoid division by zero or NaNs
    # Clip sigma to a small positive value
    sigma = np.clip(sigma, 1e-10, None)
    
    # UCB for minimization (Lower Confidence Bound of the function value)
    # We want to minimize f(x). The LCB is mu - beta * sigma.
    # We want to pick the x with the lowest LCB.
    # The task requires returning scores where HIGHER is better.
    # So we return the negative of the LCB: -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Choose beta. A larger beta encourages more exploration.
    # For 2D Ackley with 30 evals, exploration is key.
    beta = 1.0
    
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
