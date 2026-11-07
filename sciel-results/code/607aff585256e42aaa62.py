import numpy as np
from scipy.stats import norm
import time

_module_start = time.time()
_DEADLINE = 280.0  # 20s safety margin before 300s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Handle sigma = 0 case to avoid division by zero
    # If sigma is 0, the function value is deterministic.
    # If mu < f_best, EI is infinite (or very large), else 0.
    # In practice, sigma is rarely exactly 0 due to noise 1e-6.
    # But we guard against it.
    sigma_safe = np.maximum(sigma, 1e-10)
    
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Calculate EI
    # norm.cdf(z) and norm.pdf(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
