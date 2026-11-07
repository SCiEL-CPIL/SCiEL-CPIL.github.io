# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        Mean of the GP posterior at N candidate points.
    sigma : ndarray, shape (N,)
        Standard deviation of the GP posterior at N candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores; higher is better.
    """
    # Add a small epsilon to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure all values are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
