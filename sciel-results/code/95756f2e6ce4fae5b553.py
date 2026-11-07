"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Log-Probability of Improvement
    with a Hypervolume-Inspired multi-objective surrogate.
    
    We treat (objective_value, cost) as a bi-objective problem where we want to
    minimize both. The hypervolume improvement (HVI) surrogate encourages points
    that are either:
      1. Significantly better than f_best (exploitation), OR
      2. Have lower cost than the current candidate set (efficiency)
    
    Formula: Score = log(1 + PI) * (sigma^gamma) / (cost^delta) * (1 + hvi_bonus)
    
    - log(1 + PI) smoothly saturates exploitation
    - sigma^gamma with gamma > 1 emphasizes high-uncertainty regions
    - cost^delta with delta < 1 softens the cost penalty
    - hvi_bonus rewards points that dominate the "reference point" in the 
      (value, cost) space, providing a principled multi-objective signal
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 310.0:
        # Fallback: simple sigma-based score normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        fallback = np.maximum(sigma_arr, 1e-8) / np.maximum(cost_arr, 1e-8)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Clip PI to avoid log(0) issues
    pi_clipped = np.clip(pi, 1e-10, 1.0 - 1e-10)
    
    # Parameters
    gamma = 1.2   # Uncertainty exponent (> 1 to emphasize high-uncertainty regions)
    delta = 0.5   # Cost exponent (< 1 to soften cost penalty)
    
    # Log-barrier on PI: smooth, bounded, encourages moderate improvement probability
    log_pi = np.log1p(pi_clipped)
    
    # Uncertainty term: sigma^gamma
    sigma_term = np.power(sigma_safe, gamma)
    
    # Cost term: 1 / cost^delta
    cost_term = np.power(cost_safe, -delta)
    
    # Base score
    base_score = log_pi * sigma_term * cost_term
    
    # Hypervolume Improvement (HVI) surrogate:
    # We want to encourage points that are better than f_best AND have lower cost
    # than the average cost of candidates. This creates a multi-objective pressure.
    
    # Reference point for HVI: (f_best, mean_cost)
    # A point (mu, cost) "dominates" the reference if mu < f_best AND cost < mean_cost
    # HVI contribution is proportional to the area of the dominated rectangle
    
    mean_cost = np.mean(cost_safe)
    
    # For each candidate, compute the "hypervolume improvement" over the reference
    # rect_width = max(0, f_best - mu)  # improvement in objective
    # rect_height = max(0, mean_cost - cost)  # improvement in cost
    # hvi = rect_width * rect_height
    
    # But we want a smooth, normalized version. Let's use:
    # hvi_bonus = (f_best - mu) * (mean_cost - cost) when both are positive
    # This is zero when either condition is not met
    
    obj_improvement = np.maximum(f_best - mu, 0.0)
    cost_improvement = np.maximum(mean_cost - cost_safe, 0.0)
    
    # Normalize by scale to keep the bonus in a reasonable range
    obj_scale = np.std(mu) + 1e-8
    cost_scale = np.std(cost_safe) + 1e-8
    
    hvi_raw = (obj_improvement / obj_scale) * (cost_improvement / cost_scale)
    
    # Scale the HVI bonus to have a similar magnitude as the base score
    # The base score is roughly in [0, 2] range (log(2) * sigma * cost_term)
    # Let's scale HVI to be at most ~0.5x the base score magnitude
    hvi_bonus = 0.5 * hvi_raw
    
    # Combine: additive HVI bonus that encourages multi-objective improvement
    scores = base_score + hvi_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
