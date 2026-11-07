"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Z = (f_best - mu) / sigma
    # We want higher score to mean "evaluate next"
    
    Z = (f_best - mu) / sigma
    
    # Compute EI
    ei = (f_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # EI is always non-negative, and higher means more expected improvement
    return ei
# EVOLVE-BLOCK-END
