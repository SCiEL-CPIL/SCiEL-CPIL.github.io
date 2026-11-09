"""Acquisition function for Bayesian optimization of Beale 2D using Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        EI scores for each candidate. Higher scores indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF of standard normal, phi is PDF of standard normal
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure scores are finite
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ei_scores
# EVOLVE-BLOCK-END
