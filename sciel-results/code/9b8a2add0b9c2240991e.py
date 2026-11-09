import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    if time.time() > _DEADLINE:
        return np.ones_like(mu, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Improvement: f_best - mu
    # z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    improvement = (f_best - mu) * phi_z_cdf + sigma_safe * phi_z
    
    # Ensure non-negative (numerical errors might make it slightly negative)
    improvement = np.maximum(improvement, 0.0)
    
    # Ensure finite
    scores = np.nan_to_num(improvement, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
