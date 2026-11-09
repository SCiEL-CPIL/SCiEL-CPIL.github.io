"""Rewritten program using a UCB-based acquisition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.
    
    Uses Upper Confidence Bound (UCB) with cost-aware weighting.
    UCB balances exploitation (low mu) and exploration (high sigma).
    The score is normalized by the cost of evaluation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions of the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions of the GP
    f_best : float
        Best observed value so far (minimum)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB parameter: controls exploration vs exploitation
    # Higher beta means more exploration.
    # For Hartmann 3D, which is multi-modal, a higher beta is often beneficial
    # to escape local minima.
    beta = 2.0
    
    # UCB score for minimization: -mu + beta * sigma
    # We want to maximize this value.
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost
    # Since cost is in [1, 2], this mildly penalizes expensive points.
    score = ucb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
