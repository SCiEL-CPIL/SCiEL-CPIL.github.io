# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Lower Confidence Bound (LCB) for minimization.
    Score = -mu + beta * sigma.
    This encourages exploration of high-uncertainty regions and exploitation
    of low-mean regions.
    """
    # Deadline guard
    if time.time() - _START_TIME > 320:
        # Return a safe, finite score. 
        # If we are close to timeout, we should return something that doesn't crash.
        # Returning zeros or a simple heuristic is safe.
        return np.zeros_like(mu)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # So we maximize -mu + beta * sigma
    # Using beta = 2.0 for strong exploration
    beta = 2.0
    score = -mu + beta * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
