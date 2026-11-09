"""Rewritten acquisition function using Cost-Weighted Upper Confidence Bound (UCB)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Ensure sigma is positive
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB parameter: controls exploration vs exploitation
    # Beta=2.0 is a standard choice for balancing exploration and exploitation
    beta = 2.0
    
    # UCB score for minimization: maximize -mu + beta * sigma
    # This favors low predicted values (exploitation) and high uncertainty (exploration)
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware scoring: UCB per unit cost
    cost_safe = np.maximum(cost, eps)
    
    scores = ucb_scores / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
