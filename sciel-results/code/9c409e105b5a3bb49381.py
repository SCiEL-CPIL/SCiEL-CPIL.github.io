import numpy as np
import time

_DEADLINE = 330.0
_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _check_deadline():
    return _get_elapsed() < _DEADLINE - 5.0

def acquisition(mu, sigma, f_best):
    n = len(mu)
    
    if not _check_deadline():
        # Fallback to simple UCB
        return -mu + 2.0 * sigma

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta
    # If uncertainty is high, explore more.
    # If uncertainty is low, exploit more.
    med_sigma = np.median(sigma_safe)
    
    # Scale beta between 1.0 and 3.0
    # If med_sigma is 0.5, beta is 2.0
    # If med_sigma is 0.0, beta is 1.0
    # If med_sigma is 1.0, beta is 3.0
    beta = 1.0 + 2.0 * np.clip(med_sigma / 0.5, 0.0, 1.0)
    
    # UCB Score for minimization: -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
