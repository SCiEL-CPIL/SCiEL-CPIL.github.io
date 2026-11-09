import numpy as np
from scipy.special import erf, exp
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: Pure exploitation (LCB with beta=0)
        return -mu

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) using erf
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    cdf_z = 0.5 * (1.0 + erf(z / sqrt2))
    
    # Compute phi(z)
    # phi(z) = (1 / sqrt(2*pi)) * exp(-z^2 / 2)
    pdf_z = (1.0 / np.sqrt(2.0 * np.pi)) * exp(-0.5 * z * z)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
