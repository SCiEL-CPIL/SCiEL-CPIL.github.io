import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    Uses Expected Improvement (EI) for minimization.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 10.0:
        # Fallback: simple EI
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    # Improvement = f_best - f(x)
    # E[Improve] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI is always non-negative.
    # Higher EI means higher expected improvement, so higher score.
    
    # Handle numerical instability (e.g., if sigma is very small and mu < f_best, z is large positive)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
