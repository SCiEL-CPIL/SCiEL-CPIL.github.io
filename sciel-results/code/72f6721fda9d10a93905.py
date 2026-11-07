"""
Cost-aware acquisition function for Bayesian optimization (shekel 4d).
Uses a Cost-Aware UCB (Upper Confidence Bound) approach.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare before the hard limit

def _get_time_remaining():
    """
    Calculate time remaining before the hard 330s limit.
    """
    total_elapsed = time.time() - _start_time
    return 330.0 - total_elapsed

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores using Cost-Aware UCB.
    
    Args:
        mu: Mean of the GP prediction (shape: 2000,)
        sigma: Std deviation of the GP prediction (shape: 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point (shape: 2000,)
        
    Returns:
        scores: Acquisition scores (shape: 2000,), higher is better.
    """
    # Deadline Guard
    if _get_time_remaining() < _DEADLINE_MARGIN:
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # --- Main Algorithm: Cost-Aware UCB ---
    
    # Ensure sigma and cost are safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f. So we look for low mu and high sigma.
    # The "confidence bound" on the minimum is mu - beta * sigma.
    # The acquisition function should be high when this bound is low.
    # Acq = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost.
    # Alternatively, relative to f_best:
    # Acq = (f_best - mu + beta * sigma) / cost.
    # This combines exploitation (f_best - mu) and exploration (beta * sigma).
    
    # Beta is a hyperparameter controlling the exploration-exploitation trade-off.
    # A value around 2.0 is a standard starting point for BO.
    beta = 2.0
    
    ucb_score = (f_best - mu + beta * sigma_safe) / cost_safe
    
    # Ensure the score is non-negative to avoid penalizing points that are 
    # worse than f_best and have low uncertainty.
    # However, UCB can be negative if mu is much larger than f_best + beta*sigma.
    # In BO, we usually take max(0, Acq) or just use the raw score.
    # Using max(0, ...) prevents selecting points that are definitely worse.
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # Ensure finite output
    scores = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
