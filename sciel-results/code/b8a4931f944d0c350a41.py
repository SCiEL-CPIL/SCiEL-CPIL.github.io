"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    Returns scores where higher means evaluate next.
    """
    # Ensure sigma is positive and non-zero for division
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score for EI
    # For minimization, improvement is (f_best - mu)
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Clip Z to avoid numerical issues in extreme tails
    Z_clipped = np.clip(Z, -10, 10)
    
    Phi = norm.cdf(Z_clipped)
    phi = norm.pdf(Z_clipped)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure finiteness
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
# EVOLVE-BLOCK-END
