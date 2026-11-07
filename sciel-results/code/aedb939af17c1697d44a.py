"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    Uses a robust combination of Expected Improvement and UCB exploration,
    with cost-awareness that adapts to the current cost landscape.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) for exploration - lower is better
    # Use adaptive kappa based on uncertainty
    # In early stages, we want more exploration; later, more exploitation
    # Estimate "stage" from the spread of f_best relative to mu
    mu_mean = np.mean(mu)
    mu_std = np.std(mu) + 1e-8
    f_best_rel = (f_best - mu_mean) / mu_std
    
    # kappa decreases as we get closer to the minimum (f_best_rel decreases)
    # f_best_rel is positive when f_best > mean(mu), meaning we haven't found good values yet
    kappa_base = 2.0
    kappa_decay = 0.5
    kappa = kappa_base / (1.0 + 0.1 * np.abs(f_best_rel))
    kappa = np.clip(kappa, 0.5, 2.0)
    
    # LCB for minimization: mu - kappa * sigma (lower is better)
    lcb = mu - kappa * sigma_safe
    # Negate for maximization
    lcb_score = -lcb
    
    # Cost-awareness: we want to prefer cheaper points, but not too aggressively
    # The cost is in [1, 2] (since x scaled to [0,1], mean is in [0,1], cost = 1 + mean)
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-8
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Cost penalty: use a power to control aggressiveness
    # Higher power = stronger preference for cheap points
    # Use moderate power to balance exploration and cost
    cost_power = 1.5
    cost_factor = np.power(cost_norm + 0.1, cost_power)
    # Ensure cost_factor is in a reasonable range
    cost_factor_min = np.min(cost_factor)
    cost_factor_max = np.max(cost_factor)
    cost_factor_range = cost_factor_max - cost_factor_min + 1e-8
    cost_factor_norm = (cost_factor - cost_factor_min) / cost_factor_range
    
    # Invert: cheaper points should have higher score
    cost_preference = 1.0 - cost_factor_norm
    
    # Normalize EI and LCB scores
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-8
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    # Combine: 60% EI, 40% LCB, with cost preference
    # Use a weighted sum with cost as a multiplicative factor
    ei_weight = 0.6
    lcb_weight = 0.4
    cost_weight = 0.3
    
    base_score = ei_weight * ei_norm + lcb_weight * lcb_norm
    # Apply cost preference as a boost
    score = base_score * (1.0 + cost_weight * cost_preference)
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety: ensure no NaN or Inf
    if not np.all(np.isfinite(score)):
        score = np.zeros_like(score)
    
    return score
# EVOLVE-BLOCK-END
