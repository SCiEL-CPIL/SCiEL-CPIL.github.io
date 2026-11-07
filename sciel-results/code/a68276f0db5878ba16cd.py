"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE_MARGIN = 1.0  # seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline to avoid timeout
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a safe fallback: points with low cost get higher scores
        return 1.0 / np.maximum(cost, 1e-9)
    
    # Expected improvement per unit cost.
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    scores = ei / np.maximum(cost, 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
