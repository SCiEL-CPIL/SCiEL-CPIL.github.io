"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_START_TIME = None


def _get_remaining_time():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 325.0  # 5 second safety margin
    return max(0.0, _DEADLINE - time.time())


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement with a small safety margin.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    global _DEADLINE, _START_TIME
    
    # If this is the first call, initialize the deadline
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 325.0  # 5 second safety margin before 330s limit
    
    # Compute Expected Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
