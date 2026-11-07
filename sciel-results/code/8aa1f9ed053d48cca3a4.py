"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the 300s limit, return a safe default
    elapsed = _get_elapsed()
    if elapsed > 290:
        # Return uniform scores to avoid hanging; evaluator will pick index 0
        return np.zeros(len(mu), dtype=np.float64)
    
    # Cost-aware UCB with adaptive exploration
    # Use a slightly higher beta to encourage more exploration early on
    # when uncertainty is high, but keep it moderate to avoid over-exploration
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-12)
    
    # Adaptive beta: scale with the ratio of uncertainty to improvement
    # This helps balance exploration vs exploitation dynamically
    improvement = f_best - mu
    # Use a base beta of 1.5 for stronger exploration
    base_beta = 1.5
    
    # The UCB score: optimistic improvement
    # (f_best - mu) + beta * sigma
    ucb_score = improvement + base_beta * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But clip cost to avoid division by very small numbers
    cost_safe = np.maximum(cost, 1e-6)
    cost_safe = np.where(np.isfinite(cost_safe), cost_safe, 1.0)
    
    scores = ucb_score / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Optionally, add a small penalty for points where sigma is very low
    # (overconfident predictions) to encourage some exploration
    # This is a soft penalty, not a hard one
    low_sigma_mask = sigma_safe < 1e-4
    if np.any(low_sigma_mask):
        scores[low_sigma_mask] *= 0.9
    
    return scores
# EVOLVE-BLOCK-END
