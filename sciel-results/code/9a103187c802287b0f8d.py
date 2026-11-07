"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu

    # Thompson Sampling acquisition function
    # Draw a single random sample from the posterior: theta = mu + sigma * Z
    # where Z is standard normal noise
    # Score candidates based on the negative of this sample (-theta)
    # This implicitly balances exploration and exploitation through stochastic sampling
    
    # Generate standard normal noise for each candidate
    z = np.random.standard_normal(len(mu))
    
    # Compute the sampled values
    theta = mu + sigma * z
    
    # Score is the negative of the sample (higher score = lower predicted value)
    scores = -theta
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
