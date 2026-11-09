"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    n = len(mu)
    if n == 0:
        return np.array([])

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we are running too long, return greedy scores (-mu)
    elapsed = time.time() - _start_time
    if elapsed > 290:
        return -mu

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Z = (f_best - mu) / sigma
    
    Z = (f_best - mu) / sigma
    
    # Compute CDF and PDF
    # Using scipy.stats.norm
    cdf_Z = norm.cdf(Z)
    pdf_Z = norm.pdf(Z)
    
    ei = (f_best - mu) * cdf_Z + sigma * pdf_Z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
