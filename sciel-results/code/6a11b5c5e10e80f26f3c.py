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
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean
    z = (f_best - mu) / sigma_safe
    
    # Standard Normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
