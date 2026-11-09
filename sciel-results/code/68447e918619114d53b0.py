"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # Higher score means more expected improvement, so evaluate next.
    
    # Ensure sigma is not zero to avoid division errors
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate Z-score for improvement
    # Improvement is f_best - f(x).
    # Z = (f_best - mu) / sigma
    Z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Phi is the standard normal CDF
    # phi is the standard normal PDF
    EI = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Ensure finite scores
    EI = np.where(np.isfinite(EI), EI, 0.0)
    
    return EI
# EVOLVE-BLOCK-END
