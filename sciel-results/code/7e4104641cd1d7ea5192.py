"""Cost-aware acquisition function for Bayesian optimization (Levy 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Safety margin: return something valid if we're running out of time
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: just return -mu / cost to be safe
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-10)

    # Safe cost (avoid division by zero)
    safe_cost = np.maximum(cost, 1e-6)

    # Expected Improvement
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -30, 30)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware score: EI per unit cost
    score = ei / safe_cost

    # Add a small exploration bonus: prefer points with higher uncertainty
    # that are also cheap, to encourage exploration in low-cost regions
    # This helps escape local optima while respecting the cost constraint
    exploration_bonus = 0.02 * sigma / safe_cost

    total_score = score + exploration_bonus

    # Ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)

    return total_score
# EVOLVE-BLOCK-END
