"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Calculates the expected reduction in the best value found so far.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate the normalized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    # Replace non-finite values with 0 (no improvement expected)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
