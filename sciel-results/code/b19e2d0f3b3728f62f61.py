"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses a combination of Expected Improvement and Upper Confidence Bound
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher scores indicate
        more promising candidates to evaluate next.
    """
    # Handle edge cases where sigma might be very small or zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute the z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound with adaptive kappa
    # Use a moderate exploration parameter
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Combine EI and UCB with weighted average
    # Normalize both to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min + 1e-10
    ucb_range = ucb_max - ucb_min + 1e-10
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination: 60% EI, 40% UCB
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    return score
# EVOLVE-BLOCK-END
