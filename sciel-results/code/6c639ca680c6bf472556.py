"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)"""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Upper Confidence Bound (UCB) for minimization.
    
    The score is based on the potential improvement (f_best - mu) plus an exploration
    bonus (beta * sigma), all divided by the evaluation cost.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    n = len(mu)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive and finite
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB parameter for exploration-exploitation trade-off
    # For minimization, we want to maximize (f_best - mu + beta * sigma)
    beta = 2.0
    
    # Calculate the UCB potential improvement
    # If mu is low and sigma is high, this value is high.
    # f_best is the current best observed value.
    ucb_score = f_best - mu + beta * sigma_safe
    
    # We only care about points that have potential to improve
    # Clip negative scores to 0
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # Normalize by cost to get improvement per unit cost
    score = ucb_score / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Clip to reasonable range
    score = np.clip(score, 0.0, 1e6)
    
    return score
# EVOLVE-BLOCK-END
