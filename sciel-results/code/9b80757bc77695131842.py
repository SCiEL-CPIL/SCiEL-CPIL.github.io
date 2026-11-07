"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _deadline:
        return -mu
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma is strictly positive for division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of standard normal
    phi_z = norm.pdf(z)
    cdf_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: We use sigma (original) for the second term to preserve scale, 
    # but sigma_safe for z calculation. If sigma was 0, EI should be max(f_best-mu, 0).
    
    ei = (f_best - mu) * cdf_z + sigma_safe * phi_z
    
    # Correction for points where sigma was very small (effectively 0)
    # If sigma < 1e-8, the model is certain.
    # EI should be max(f_best - mu, 0)
    mask_low_sigma = sigma < 1e-8
    if np.any(mask_low_sigma):
        ei[mask_low_sigma] = np.maximum(f_best - mu[mask_low_sigma], 0.0)
        
    # Ensure finite scores
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
