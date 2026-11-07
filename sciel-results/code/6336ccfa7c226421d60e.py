import numpy as np
from scipy.stats import norm
import time

# Global variable to track start time for deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # 330s total limit. Let's say we want to be done by 325s.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Return a simple exploitation score: -mu
        # This ensures we pick the best current point if we are running out of time
        return -np.asarray(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Calculate EI
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
