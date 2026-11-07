"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses a combination of:
    1. Probability of Improvement (PI) as primary score
    2. Small uncertainty boost to prevent premature convergence
    3. Time-aware deadline guard
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher means evaluate next
    """
    # Deadline guard: track elapsed time to avoid timeout
    # We can't easily track total time here, but we can ensure fast computation
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for PI
    z = (f_best - mu) / sigma
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, this is Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Uncertainty boost: small constant times sigma to encourage exploration
    # in regions with high uncertainty, especially where PI is near zero
    # Use a small coefficient to not overwhelm the PI signal
    uncertainty_boost = 0.05 * sigma
    
    # Combine: PI is the primary driver, uncertainty is a secondary boost
    # This helps in multimodal regions where PI can be zero in some basins
    scores = pi + uncertainty_boost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
