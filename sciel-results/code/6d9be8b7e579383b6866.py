"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    Computes the expected amount by which the function value at x
    will improve upon the current best observed value f_best.
    """
    # Clip sigma to avoid division by zero and ensure positivity
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z is positive (improvement likely)
    # If mu > f_best, z is negative (improvement unlikely)
    z = (f_best - mu) / sigma
    
    # Phi(z) is the standard normal CDF
    # phi(z) is the standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite
    # Replace non-finite values with a very low score (low priority)
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
