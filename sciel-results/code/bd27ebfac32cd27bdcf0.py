"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety deadline
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s

# Global iteration counter for adaptive beta
_ITER_COUNT = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier ranking of LCB improvement.
    Returns 2000 finite scores (higher = evaluate next).
    """
    global _ITER_COUNT
    
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware LCB
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        beta = 2.0
        lcb = mu - beta * sigma_safe
        imp = f_best - lcb
        return imp / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    _ITER_COUNT += 1

    # Ensure safety: sigma and cost must be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Adaptive Beta: Start high for exploration, decay slowly
    # In 8D, we need significant exploration.
    beta = 3.0 / (1.0 + 0.05 * _ITER_COUNT)

    # Compute LCB-based improvement
    # For minimization, optimistic value is mu - beta*sigma
    # Improvement is f_best - optimistic_value
    lcb = mu - beta * sigma_safe
    improvement = f_best - lcb

    # Pareto Ranking:
    # Ideal point: Min Cost, Max Improvement
    # We normalize Cost and Improvement to [0, 1] and compute distance to (0, 1)
    
    # Normalize Cost
    c_min = np.min(cost_safe)
    c_max = np.max(cost_safe)
    c_range = c_max - c_min
    if c_range < 1e-9:
        c_norm = np.zeros_like(cost_safe)
    else:
        c_norm = (cost_safe - c_min) / c_range

    # Normalize Improvement
    i_min = np.min(improvement)
    i_max = np.max(improvement)
    i_range = i_max - i_min
    if i_range < 1e-9:
        i_norm = np.ones_like(improvement) # If all same, treat as max? Or 0.5?
        # If all improvements are same, we should fall back to cost only or random.
        # But if i_range is 0, it means GP is perfectly flat or beta is 0.
        # Let's set i_norm to 0.5 to be neutral, so cost dominates.
        i_norm = np.full_like(improvement, 0.5)
    else:
        i_norm = (improvement - i_min) / i_range

    # Distance to ideal point (c_norm=0, i_norm=1)
    # D = sqrt(c_norm^2 + (1 - i_norm)^2)
    # We want to MINIMIZE distance, so we MAXIMIZE -Distance
    # Or simply return 1 - D (scaled to be positive if possible)
    
    dist = np.sqrt(c_norm**2 + (1.0 - i_norm)**2)
    
    # Scores: Higher is better.
    # dist is in [0, sqrt(2)].
    # We can just return -dist.
    # Or to make it more "score-like", return 1 - dist.
    scores = 1.0 - dist

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Replace -inf with a very small number to ensure valid output? 
    # The task says "Return 2000 finite scores".
    # So we must not return -inf.
    min_score = np.min(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else -1.0
    scores = np.where(np.isfinite(scores), scores, min_score - 1.0)

    return scores
# EVOLVE-BLOCK-END
