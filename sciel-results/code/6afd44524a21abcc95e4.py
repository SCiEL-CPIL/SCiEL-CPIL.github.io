"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization (minimization).
    
    EI is a robust choice for minimizing functions like Michalewicz 2D.
    It balances exploitation (low mu) and exploration (high sigma) by calculating
    the expected amount of improvement over the current best value f_best.
    
    Formula for minimization:
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    Phi is the standard normal CDF, phi is the standard normal PDF.
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # Fallback to pure exploitation if time is running out
        return -mu

    n = len(mu)
    
    # Sanitize inputs to avoid division by zero or negative variance issues
    # sigma should be non-negative, but numerical errors can occur
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the z-score: (f_best - mu) / sigma
    # We want to find points where mu is significantly lower than f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) is norm.cdf(z)
    # phi(z) is norm.pdf(z)
    
    # Using scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure all values are finite and non-negative (EI is theoretically >= 0)
    # In case of numerical issues, clip to 0
    ei = np.maximum(ei, 0.0)
    
    # Replace any non-finite values with 0 (least attractive)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
