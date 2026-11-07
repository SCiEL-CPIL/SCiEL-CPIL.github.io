"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_deadline = _start_time + 290.0
_step_count = 0

def acquisition(mu, sigma, f_best):
    global _step_count
    _step_count += 1
    
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    # Expected Improvement (EI) for minimization
    # We want to maximize the expected improvement over f_best
    # Improvement = max(0, f_best - Y)
    # Y ~ N(mu, sigma^2)
    
    # To avoid numerical issues with division by zero or very small sigma
    # we clip sigma to a small positive value
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best < mu, z is negative. Phi(z) is small, phi(z) is small.
    # If f_best > mu, z is positive. This is the standard EI formula.
    
    # We use scipy.stats.norm for CDF (Phi) and PDF (phi)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
