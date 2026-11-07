import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight (it's already EI, so just return it)
        pass

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # norm.cdf(z) and norm.pdf(z)
    # For large positive z, cdf -> 1, pdf -> 0. EI -> f_best - mu.
    # For large negative z, cdf -> 0, pdf -> 0. EI -> 0.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
