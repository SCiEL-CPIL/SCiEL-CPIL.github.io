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
    sigma = np.asarray(sigma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Deadline guard: if we are running too long, return greedy scores (-mu)
    # to ensure we finish within the time limit.
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Greedy: pick the point with the lowest mean prediction
        # Score = -mu (higher is better)
        return -mu

    # Expected Improvement (EI) for minimization
    # We want to maximize EI.
    # EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    
    delta = f_best - mu
    z = delta / sigma
    
    # Calculate Phi(z) and phi(z)
    # Using scipy.stats.norm for accuracy
    phi_z = norm.pdf(z)
    phi_z_cdf = norm.cdf(z)
    
    ei = delta * phi_z_cdf + sigma * phi_z
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is non-negative by definition (max(0, improvement))
    # However, due to numerical errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
