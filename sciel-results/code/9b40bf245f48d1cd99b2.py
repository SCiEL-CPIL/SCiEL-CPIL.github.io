"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 300s per eval / 330s wall-clock
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        # Fallback: return random-ish scores based on index
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # For numerical stability with large |z|
    # Phi(z) ~ 0 for z << 0, Phi(z) ~ 1 for z >> 0
    # phi(z) ~ 0 for |z| >> 0
    
    # Use scipy's norm.cdf and norm.pdf which are stable
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Check for non-finite values
    if not np.all(np.isfinite(ei)):
        ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # The evaluator picks the point with the HIGHEST score.
    # EI is higher when improvement is larger (lower mu, high sigma).
    # This is the standard acquisition function.
    
    # Check deadline - EI computation is O(N) and very fast for N=2000
    # No need for early termination here as it's a single vectorized op.
    
    return ei
# EVOLVE-BLOCK-END
