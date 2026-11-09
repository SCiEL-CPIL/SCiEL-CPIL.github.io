"""Acquisition function using Expected Improvement for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) with posterior means.
        sigma: array of shape (2000,) with posterior standard deviations.
        f_best: scalar, the lowest observed value so far.
        
    Returns:
        array of shape (2000,) with acquisition scores (higher is better).
    """
    # Deadline guard: if we are too close to the time limit, return a safe, fast result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to a simple exploitation score to ensure we return quickly
        # This case is highly unlikely given the vectorized nature of the operation
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We maximize EI, so higher scores are better.
    
    z = (f_best - mu) / sigma
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
