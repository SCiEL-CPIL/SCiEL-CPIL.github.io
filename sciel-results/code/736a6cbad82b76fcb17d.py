"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time and return early if approaching limit
_START_TIME = None


def acquisition(mu, sigma, f_best):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: return current best if approaching 330s limit
    elapsed = time.time() - _START_TIME
    if elapsed > 325.0:  # Safety margin of 5s
        # Return a simple score that favors points with low mu
        return -mu
    
    # Stable Expected Improvement calculation
    # Handle edge cases where sigma is very small
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Use stable formulations for CDF and PDF
    # For large negative z, cdf ~ 0 and pdf ~ 0
    # For large positive z, cdf ~ 1 and pdf ~ 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is standard EI for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
