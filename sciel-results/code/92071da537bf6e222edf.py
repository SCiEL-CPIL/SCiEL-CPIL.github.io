"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Uses Expected Improvement with adaptive exploration and deadline guard.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return simple score if approaching limit
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        return -mu
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration bonus: scale with uncertainty but cap it
    # Use a smaller bonus than before to be more exploitation-focused
    exploration_bonus = 0.005 * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
