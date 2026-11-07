"""Rewritten acquisition using UCB with adaptive exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB with a beta parameter that balances exploration and exploitation
    # For minimization, we want to minimize f(x), so UCB for minimization is:
    # ucb = mu - beta * sigma  (we want low mu and high sigma for exploration)
    # But typically UCB is defined for maximization. For minimization:
    # We want to find points where the lower confidence bound is low.
    # Lower Confidence Bound for minimization: LCB = mu - beta * sigma
    # We want to maximize the LCB? No, we want to minimize f, so we pick points
    # where the lower bound is as low as possible. But acquisition functions
    # return higher scores for better candidates.
    
    # Let's think carefully:
    # For minimization, a good candidate has low mu and high sigma (uncertainty).
    # The UCB for minimization can be defined as: ucb_min = mu - beta * sigma
    # We want to select the point with the lowest ucb_min.
    # But our acquisition function should return HIGHER scores for better candidates.
    # So we return: score = -ucb_min = -mu + beta * sigma
    # Or equivalently: score = beta * sigma - mu
    
    # Let's use a moderate beta value. The optimal beta depends on the problem,
    # but a common choice is around 1.0 to 3.0. Let's use beta = 1.5 as a start.
    beta = 1.5
    
    scores = beta * sigma - mu
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
