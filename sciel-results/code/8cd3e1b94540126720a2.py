# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best):
    # Deadline guard to ensure we don't exceed time limits
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 10:
        # Return a neutral score if we are running out of time
        return np.zeros_like(mu, dtype=float)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)

    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # Higher EI means higher expected improvement over f_best.
    
    # To handle numerical stability and avoid issues with very large/small values in the normal CDF/PDF:
    # We compute the standardized variable z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    # Using scipy.stats for robustness
    # Note: norm.cdf and norm.pdf are vectorized
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    # If sigma is very small, z can be large.
    # For large positive z, Phi(z) -> 1, phi(z) -> 0. EI -> f_best - mu.
    # For large negative z, Phi(z) -> 0, phi(z) -> 0. EI -> 0.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure the output is finite to prevent invalid results
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative.
    # We return EI as the score. Higher is better.
    return ei
# EVOLVE-BLOCK-END
