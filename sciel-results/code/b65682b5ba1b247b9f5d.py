"""Rewritten program using Expected Improvement (EI) acquisition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Expected Improvement (EI) to balance exploitation and exploration.
    EI is a robust choice for multimodal functions as it considers both the
    likelihood of improvement and the magnitude of that improvement.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process posterior at candidate points (objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points (objective units).
    f_best : float
        Best (lowest) observed value so far (objective units).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    # Use a very small epsilon to avoid division by zero while preserving
    # the logic for points with very high certainty.
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    # We are minimizing, so we want f(x) < f_best
    # Improvement I(x) = max(0, f_best - f(x))
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI
    # norm.cdf is the cumulative distribution function of the standard normal
    # norm.pdf is the probability density function of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle numerical instabilities
    # EI should theoretically be >= 0, but floating point errors might make it slightly negative
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    ei = np.maximum(ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
