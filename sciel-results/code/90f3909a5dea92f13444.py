import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None

def acquisition(mu, sigma, f_best):
    global _bo_start_time
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
        
    # Deadline Guard: 330s limit, use 300s as safety margin
    if time.time() - _bo_start_time > 300:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        # Return -mu (pure exploitation) or a simple heuristic.
        return -mu

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finite scores
    scores = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
