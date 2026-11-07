"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Expected Improvement (EI).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf for extreme values
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus to encourage visiting unexplored areas
    # if the EI is very small, the sigma term can help break symmetry or
    # explore regions with high uncertainty but slightly worse mean.
    # However, pure EI is often sufficient. 
    # Let's add a tiny multiple of sigma to ensure we don't completely ignore
    # high-uncertainty points if EI is zero (which happens if mu >= f_best).
    # If mu >= f_best, EI=0. But if sigma is high, it might be worth checking.
    # A small lambda * sigma ensures this.
    lambda_explore = 0.1
    scores = ei + lambda_explore * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
