"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining expected improvement with an uncertainty bonus.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small uncertainty bonus to encourage exploration early on
    # The weight decays as f_best improves (gets closer to 0)
    # For Beale function, minimum is 0, so f_best should approach 0
    # Use a fixed small weight for exploration
    exploration_weight = 0.1
    bonus = exploration_weight * sigma_safe
    
    return ei + bonus
# EVOLVE-BLOCK-END
