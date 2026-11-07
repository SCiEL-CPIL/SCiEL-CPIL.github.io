"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: float, the best observed value so far.
    
    Returns:
    scores: array of shape (N,) with higher values indicating better candidates.
    """
    # Handle cases where sigma is very small to avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # For minimization, we want to find points where f(x) < f_best.
    # Improvement is max(f_best - f(x), 0).
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative, Phi(z) is small, phi(z) is small.
    # If mu < f_best, z is positive, Phi(z) is large, phi(z) is significant.
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
