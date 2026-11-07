"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -np.asarray(mu, dtype=np.float64)
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Thompson Sampling approach:
    # Draw a single random sample from the posterior at each candidate point.
    # theta = mu + sigma * Z, where Z ~ N(0, 1)
    # Score by -theta (since we want to minimize, lower theta is better,
    # so higher score = more likely to be the minimum)
    
    # Generate standard normal noise
    Z = np.random.standard_normal(len(mu))
    
    # Sample from posterior
    theta = mu + sigma * Z
    
    # Score: negative of the sample (higher score means more likely to be good)
    scores = -theta
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
