"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for the given candidates.
    
    Uses a cost-aware UCB with exploration that adapts to remaining budget.
    The acquisition combines:
    1. Expected improvement (exploitation)
    2. Uncertainty bonus (exploration)
    3. Cost penalty (efficiency)
    
    mu, sigma, f_best, cost: shape (2000,)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI) - standard formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # Uncertainty term - encourage exploring uncertain regions
    # Use a square root of sigma to balance exploration
    uncertainty = np.sqrt(sigma_safe)
    
    # Cost-aware combination
    # Weight EI heavily, add exploration bonus, divide by cost
    # The exploration weight adapts: more exploration when sigma is high relative to improvement
    # Use a dynamic weight based on the ratio of uncertainty to best observed improvement
    improvement_scale = max(abs(f_best), 1e-6)
    weight = 1.0 + 0.5 * (sigma_safe / improvement_scale)
    
    # Combine: weighted EI + uncertainty bonus, all divided by cost
    score = (weight * ei + 0.3 * uncertainty) / cost_safe
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
