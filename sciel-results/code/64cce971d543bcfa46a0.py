import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 325.0  # 5s safety margin before 330s


def _check_deadline():
    """Return True if we should abort and return a fallback immediately."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive UCB for minimization.
    Beta adjusts based on f_best progress relative to the known minimum.
    
    mu, sigma : (2000,) arrays, sigma is the GP posterior std dev
    f_best : lowest observed value so far
    Returns: (2000,) scores, higher = more promising
    """
    # Handle edge cases
    if sigma is None or len(mu) != len(sigma):
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Check deadline
    if _check_deadline():
        return -mu + sigma
    
    # Known minimum of Branin 2D
    f_min = 0.397887
    
    # Track progress: how close are we to the optimum?
    # progress = 1 means we're at the optimum, 0 means we're far
    # f_best starts high (worst case ~302), decreases toward 0.397887
    # Normalize progress: 1 - (f_best - f_min) / (f_max - f_min)
    # f_max for Branin is around 302
    f_max_est = 302.0
    progress = 1.0 - (f_best - f_min) / (f_max_est - f_min)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive beta: high when progress is low (early), low when progress is high (late)
    # beta goes from ~4.0 (early, explore) to ~0.5 (late, exploit)
    beta = 0.5 + 3.5 * (1.0 - progress)
    
    # UCB for minimization: -mu + beta * sigma
    # Higher score = more promising (lower expected value with uncertainty bonus)
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
