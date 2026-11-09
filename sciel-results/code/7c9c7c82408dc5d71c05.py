import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    elapsed = time.time() - _start_time
    if elapsed > 290:
        return -mu

    _call_count += 1
    
    delta = f_best - mu
    z = delta / sigma
    
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    ei = delta * cdf_z + sigma * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
