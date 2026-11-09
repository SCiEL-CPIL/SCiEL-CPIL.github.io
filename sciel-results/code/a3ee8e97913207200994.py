"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time and call count
_T0 = [None]
_CALL_COUNT = [0]


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using POI + regret lower bound.
    
    Args:
        mu: Array of GP means (2000,)
        sigma: Array of GP standard deviations (2000,)
        f_best: Best observed function value (scalar)
        cost: Array of evaluation costs (2000,)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    # Initialize timer on first call
    if _T0[0] is None:
        _T0[0] = time.time()
    
    elapsed = time.time() - _T0[0]
    _CALL_COUNT[0] += 1
    
    # Deadline guard with safety margin (330s limit, use 310s for safety)
    if elapsed > 310.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Fallback: simple exploration-based score
        score = sigma_safe / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Probability of Improvement (POI) per unit cost
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Regret lower bound: expected improvement lower bound
    # E[max(0, f_best - f(x))] >= (f_best - mu + sigma * 0.5) when f_best > mu
    # This is a linear approximation that's cheaper to compute and less exploitative
    regret_lb = np.maximum(0.0, f_best - mu + sigma_safe * 0.5)
    
    # Combine POI and regret lower bound, both cost-normalized
    # POI gives probability of improvement, regret_lb gives magnitude of potential improvement
    # Weight them to balance exploration (POI) and exploitation (regret_lb)
    
    # As f_best approaches 0, we're close to optimum - reduce exploration weight
    f_best_factor = max(0.0, 1.0 - min(f_best, 1.0))
    
    # Time-based decay: start with more exploration, reduce over time
    time_fraction = min(elapsed / 300.0, 1.0)
    
    # Exploration weight: higher early, lower late
    # Also modulated by how close we are to the optimum
    poi_weight = 0.3 * (1.0 - 0.5 * time_fraction) * (1.0 - 0.5 * f_best_factor)
    regret_weight = 1.0
    
    # Cost-normalized scores
    poi_score = poi_weight * poi / cost_safe
    regret_score = regret_weight * regret_lb / cost_safe
    
    score = poi_score + regret_score
    
    # Numerical safety
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
