"""Rewritten program using Cost-Adjusted Expected Improvement (CEI)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Upper Confidence Bound (UCB)
    # For minimization, we want to maximize: -mu + beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma).
    # We divide by cost to make it cost-aware.
    
    # Hyperparameter for exploration/exploitation trade-off
    beta = 2.5
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Compute UCB score for minimization
    # We want high scores for points likely to be low.
    # -mu rewards low mean values.
    # beta * sigma rewards high uncertainty.
    ucb_score = -mu + beta * sigma_safe
    
    # Cost adjustment
    cost_safe = np.maximum(cost, 1e-9)
    score = ucb_score / cost_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
