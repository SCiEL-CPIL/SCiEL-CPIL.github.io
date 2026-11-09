"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 330.0  # Seconds, with safety margin


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 10:  # 10s safety margin
        # Return a simple score based on low cost to avoid timeout
        return -cost
    
    # Cost-aware acquisition: Expected Improvement per unit cost
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative EI
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    scores = ei / cost
    
    # Return finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
