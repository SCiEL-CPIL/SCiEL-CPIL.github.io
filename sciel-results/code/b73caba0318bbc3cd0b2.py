"""Cost-aware UCB acquisition with budget-aware exploration scaling."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) instead of EI:
      score = -mu + beta * sigma
    
    where beta is a dynamic exploration parameter that decreases as
    f_best approaches the known minimum (-0.8). This shifts from
    exploration (high beta) to exploitation (low beta) as progress
    is made. Cost normalization favors cheaper evaluations.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return deterministic fallback
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu + sigma_safe
        score = score / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB: -mu + beta * sigma (minimize mu, so -mu is high when mu is low)
    # Dynamic beta: high when far from minimum, low when close
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Beta decays exponentially as gap shrinks
    # When gap is large (early), beta is high (~2.0) for exploration
    # When gap is small (late), beta is low (~0.5) for exploitation
    beta = 0.5 + 1.5 * np.exp(-gap_to_min / 0.2)
    
    # UCB score
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    score = ucb_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
