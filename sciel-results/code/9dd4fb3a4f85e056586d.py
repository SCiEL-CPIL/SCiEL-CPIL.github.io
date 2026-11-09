"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB with cost normalization.
    
    Uses a hybrid approach:
    1. UCB (Upper Confidence Bound) for exploration: mu + beta * sigma
    2. Cost-aware normalization: divide by cost to favor cheaper points
    3. Add a small exploitation term to ensure we still exploit promising regions
    
    The beta parameter is tuned to balance exploration/exploitation.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # UCB term: balances exploitation (mu) and exploration (sigma)
    # Higher beta = more exploration
    beta = 2.0
    
    # UCB score (higher is better for maximization, but we're minimizing)
    # For minimization, we want low mu and high sigma (uncertainty)
    # So we use: -mu + beta * sigma (negate mu since lower is better)
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    # This makes the acquisition function prefer points that are both promising
    # and cheap to evaluate
    cost_aware = ucb / cost
    
    # Add a small exploitation bonus for points close to f_best
    # This helps ensure we don't completely ignore exploitation
    # The term (f_best - mu) is positive when mu < f_best (better than best)
    # We scale it down to make it a secondary consideration
    exploitation_bonus = 0.1 * (f_best - mu) / cost
    
    # Combine: primarily UCB cost-aware, with small exploitation bonus
    score = cost_aware + exploitation_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
