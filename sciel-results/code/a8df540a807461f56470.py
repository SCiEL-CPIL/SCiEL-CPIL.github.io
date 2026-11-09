"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 320.0  # safety margin before 330s
    return _DEADLINE

def _time_left():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - evaluation costs
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Cost-Adjusted Lower Confidence Bound (LCB) formulation
    # score = mu + kappa * sigma - lambda * cost
    # This directly optimizes for the minimum observed value rather than incremental improvement
    
    # Adaptive kappa based on uncertainty relative to current best
    # When sigma is large relative to f_best, we want more exploration
    sigma_max = np.max(sigma)
    if sigma_max > 0:
        # Scale kappa inversely with the spread of f_best to balance exploration/exploitation
        kappa = 2.0
    else:
        kappa = 1.0
    
    # Base LCB score: lower is better, so we negate for "higher = better"
    lcb_score = -(mu - kappa * sigma)
    
    # Cost penalty: prefer cheaper evaluations
    # Normalize cost to make penalty meaningful
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    lambda_cost = 0.5  # Weight for cost penalty
    
    scores = lcb_score - lambda_cost * cost_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very negative finite value
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
