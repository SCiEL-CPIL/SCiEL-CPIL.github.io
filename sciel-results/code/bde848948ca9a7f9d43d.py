"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s; we keep a safety margin.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a cost-normalized Lower Confidence Bound (LCB) with dynamic exploration.
    The form is: -(mu - kappa * sigma) / cost
    
    Where kappa is dynamically set based on the current state to balance
    exploration and exploitation, with higher exploration in expensive regions.
    """
    # Check deadline - if we're close to the limit, return a safe default
    if _elapsed() > _DEADLINE - 2.0:
        # Return a simple exploitation-based score to avoid timeout
        return -(mu) / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no zeros or negatives in cost
    cost = np.maximum(cost, 1e-6)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Dynamic kappa: base exploration bonus
    # In regions with high cost, we want less exploration (higher kappa penalizes less)
    # In regions with low cost, we want more exploration
    # kappa scales inversely with cost to make exploration more aggressive in cheap regions
    
    # Base kappa value
    kappa_base = 1.0
    
    # Additional exploration based on how good f_best is relative to the data
    # If f_best is very good (low), we can afford more exploitation
    # If f_best is poor, we need more exploration
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # Normalized position of f_best in the observed range
    f_best_norm = np.clip((f_best - mu_min) / mu_range, 0.0, 1.0)
    
    # When f_best is close to the minimum observed, increase exploitation
    # When f_best is far from the minimum, increase exploration
    kappa_dynamic = kappa_base * (1.0 + 0.5 * f_best_norm)
    
    # Compute LCB score: lower predicted value is better, higher uncertainty is better
    # We negate because we want higher scores = evaluate next
    # score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost
    
    score = (kappa_dynamic * sigma - mu) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
