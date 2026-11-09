import numpy as np
from scipy.stats import norm
import time

# Track call index and start time for deadline guard
_call_count = [0]
_start_time = [None]

def acquisition(mu, sigma, f_best):
    _call_count[0] += 1
    step = _call_count[0]
    
    # Deadline guard: track elapsed time
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    # Safety margin: return early if we're getting close to 330s
    if elapsed > 320:
        # Return exploitation-based scores as fallback
        sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -1e10)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Thompson Sampling: sample from the posterior
    # sampled = mu + sigma * z, where z ~ N(0,1)
    # Return -sampled as scores (higher = better for minimization)
    z = np.random.standard_normal(mu.shape)
    sampled = mu + sigma_safe * z
    scores = -sampled
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
