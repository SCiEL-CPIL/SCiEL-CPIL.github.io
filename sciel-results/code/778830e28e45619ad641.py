"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a 2D RBF mixture.
    
    Uses Expected Improvement (EI) for minimization.
    
    Parameters:
        mu: array of shape (2000,), posterior mean at candidates
        sigma: array of shape (2000,), posterior std at candidates
        f_best: scalar, best (lowest) observed value so far
    
    Returns:
        array of shape (2000,), scores (higher = more promising to evaluate next)
    """
    # Deadline guard: if we're close to the time limit, return a simple score quickly
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: return something finite and simple
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # For a Gaussian posterior f(x) ~ N(mu(x), sigma(x)^2):
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    
    # Clip mu to avoid extreme values causing numerical issues
    mu_safe = np.clip(mu, -1e6, 1e6)
    
    # Compute the standardized improvement
    z = (f_best - mu_safe) / sigma_safe
    
    # Compute EI
    # Use scipy.stats.norm for Phi and phi
    # Phi(z) = CDF of standard normal at z
    # phi(z) = PDF of standard normal at z
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI can be very small or zero; clip to non-negative for safety
    ei = np.maximum(ei, 0.0)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
