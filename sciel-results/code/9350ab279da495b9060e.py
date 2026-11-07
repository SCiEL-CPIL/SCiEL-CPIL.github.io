"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a hybrid of Expected Improvement and LCB,
    with a Pareto-inspired normalization to balance improvement and cost.
    
    Args:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std predictions
        f_best: float, best observed (lowest) value
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        scores: (2000,) finite scores, higher = pick next
    """
    if time.time() > _DEADLINE:
        # Fallback: prefer cheap points with some uncertainty
        return 1.0 / (np.maximum(cost, 1e-9))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB score (optimistic improvement)
    # beta=2.0 gives ~95% confidence exploration
    beta = 2.0
    lcb = f_best - mu + beta * sigma_safe
    
    # Combine EI and LCB: use a weighted blend
    # EI is more conservative, LCB is more exploratory
    # For Rastrigin (multimodal), we want more exploration early
    # Use 0.5 * EI + 0.5 * LCB for balanced behavior
    combined = 0.5 * ei + 0.5 * lcb
    combined = np.maximum(combined, 0.0)
    
    # Pareto-frontier inspired normalization:
    # Instead of simply dividing by cost, we normalize each metric
    # (improvement signal and cost efficiency) to [0, 1] and combine
    # This prevents cost from arbitrarily dominating the improvement signal
    
    # Normalize combined improvement signal to [0, 1]
    combined_min = np.min(combined)
    combined_max = np.max(combined)
    combined_range = combined_max - combined_min + 1e-12
    norm_improvement = (combined - combined_min) / combined_range
    
    # Normalize cost efficiency (1/cost) to [0, 1]
    inv_cost = 1.0 / cost_safe
    inv_cost_min = np.min(inv_cost)
    inv_cost_max = np.max(inv_cost)
    inv_cost_range = inv_cost_max - inv_cost_min + 1e-12
    norm_cost_eff = (inv_cost - inv_cost_min) / inv_cost_range
    
    # Combine the two normalized metrics
    # Use a product to require both good improvement AND good cost efficiency
    # This is a multiplicative Pareto selection: both metrics must be decent
    # Use a soft product with a blending parameter
    # score = norm_improvement * norm_cost_eff^alpha
    # alpha < 1 softens the cost penalty
    
    alpha = 0.5  # soft cost penalty
    scores = norm_improvement * (norm_cost_eff ** alpha)
    
    # Add a small base score to ensure points with zero normalized improvement
    # (the worst points) still have a finite, non-negative score
    # This prevents -inf issues and keeps the score landscape smooth
    scores = 1e-6 + scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
