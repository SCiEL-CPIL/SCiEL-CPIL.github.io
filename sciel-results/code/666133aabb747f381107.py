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
    Acquisition function: Expected Improvement (EI) for minimization,
    with a small adaptive LCB blend for early exploration.
    
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
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    ei = z * norm.pdf(z) + norm.cdf(z)
    ei = ei * sigma
    
    # Adaptive LCB blend for early exploration
    # Use a count-based heuristic: track how many times we've been called
    if not hasattr(acquisition, 'count'):
        acquisition.count = 0
    acquisition.count += 1
    t = 5 + acquisition.count  # 5 initial points + current call index
    
    # Beta for LCB: decays slowly
    beta = 2.0 / np.log(t + 1)
    lcb = -mu + beta * sigma
    
    # Blend: weight EI more as we progress, keep some exploration early
    # The blend weight for EI increases with t
    # w_ei goes from ~0.5 early to ~1.0 later
    w_ei = min(1.0, 0.5 + 0.1 * np.log(t + 1))
    w_lcb = 1.0 - w_ei
    
    # Combine: higher score = more promising
    scores = w_ei * ei + w_lcb * lcb
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
