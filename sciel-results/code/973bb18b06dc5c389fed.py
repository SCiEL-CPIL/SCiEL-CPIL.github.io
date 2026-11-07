"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: UCB with linear cost penalty.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: ensure we finish well within 330s
    # The function is called per iteration; keep it fast.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # UCB: balance exploitation (low mu) and exploration (high sigma)
    # beta controls exploration-exploitation tradeoff
    beta = 0.5
    
    # UCB score: higher is better for maximization, but we're minimizing
    # So we want low mu and high sigma
    # ucb = -mu + beta * sigma  (higher = more promising for minimization)
    ucb = -mu + beta * sigma_safe
    
    # Cost penalty: divide by cost to prefer cheap evaluations
    # Use linear cost scaling (cost^1) for strong preference for cheap points
    cost_safe = np.maximum(cost, 1e-9)
    
    # Final score: UCB per unit cost
    scores = ucb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
