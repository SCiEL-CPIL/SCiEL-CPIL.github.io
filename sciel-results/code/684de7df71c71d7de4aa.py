"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB acquisition: reward uncertainty (sigma) and low predicted values
    # normalized by cost to prefer cheap evaluations
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # UCB: we want to minimize, so we prefer low mu and high sigma
    # UCB for minimization: -mu + beta * sigma
    # Higher score = better (we want to evaluate points with low mu and high uncertainty)
    
    # Beta controls exploration-exploitation tradeoff
    # For Styblinski-Tang with multiple local minima, we need sufficient exploration
    beta = 1.0
    
    # UCB score: negative mu (lower is better) plus exploration bonus
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize by cost to prefer cheaper evaluations
    # Cost is in [1, 2] based on the problem description
    score = ucb_score / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
