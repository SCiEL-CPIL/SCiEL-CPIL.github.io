"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    
    Uses a hybrid approach: Expected Improvement with an adaptive exploration bonus
    that accounts for the uncertainty in the GP prediction.
    
    Parameters:
    -----------
    mu : ndarray of shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : ndarray of shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray of shape (2000,)
        Acquisition scores, higher values are preferred.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration in early stages
    # The bonus is proportional to sigma, which naturally decays as the GP
    # becomes more certain
    # Use a moderate exploration coefficient
    exploration_coef = 0.5
    scores = ei + exploration_coef * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
