"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu (np.ndarray): Mean of the Gaussian process posterior, shape (N,).
    sigma (np.ndarray): Standard deviation of the Gaussian process posterior, shape (N,).
    f_best (float): The best (lowest) observed function value so far.
    
    Returns:
    scores (np.ndarray): EI scores, shape (N,). Higher values indicate better candidates.
    """
    # Avoid division by zero and handle very small sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate the improvement: f_best - mu
    # If mu < f_best, the expected improvement is positive.
    improvement = f_best - mu
    
    # Z-score
    z = improvement / sigma_safe
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal distribution.
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
