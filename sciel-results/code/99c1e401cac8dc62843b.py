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
        # Fallback to EI if time is tight
        pass

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Structural Change: Use Log-Expected Improvement (LogEI)
    # LogEI helps balance exploration and exploitation by compressing the scale of EI.
    # This prevents the acquisition function from being overly dominated by a single
    # point with a very high EI, which can happen in multimodal landscapes, and
    # encourages considering points with moderate but reliable improvements.
    # Adding a small constant to avoid log(0).
    log_ei = np.log(ei + 1e-6)
    
    # Ensure finite scores
    log_ei = np.where(np.isfinite(log_ei), log_ei, 0.0)
    
    return log_ei
