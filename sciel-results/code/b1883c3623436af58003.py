"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses Expected Improvement (EI) for minimization.
    EI(x) = E[max(f_best - f(x), 0)]
    For a GP posterior with mean mu and std sigma:
    z = (f_best - mu) / sigma
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple, fast, safe score if time is critical
        # Returning EI is a reasonable fallback as it's a standard baseline
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        ei = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
        return np.where(np.isfinite(ei), ei, 0.0)

    # Numerical safety for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute standard normal score z
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI (Expected Improvement)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    ei_score = (f_best - mu) * ndtr(z) + sigma_safe * phi_z
    
    # Ensure all scores are finite
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    
    return ei_score
# EVOLVE-BLOCK-END
