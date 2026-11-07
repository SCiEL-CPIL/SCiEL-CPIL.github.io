# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a combination of expected improvement (EI) and a penalty for high cost,
    with an additional exploration bonus to avoid local optima.
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed value so far
    cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: normalize cost to [0, 1] for consistent scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-10:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Cost penalty: prefer cheaper points
    # Use a soft penalty that decreases with cost
    cost_penalty = 1.0 / (cost_norm + 1.0)
    
    # Exploration bonus: encourage exploring uncertain regions
    # Normalize sigma to [0, 1]
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-10:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus: weighted by uncertainty
    exploration_bonus = 0.1 * sigma_norm
    
    # Combine: EI with cost penalty and exploration bonus
    # Weight EI heavily, but allow exploration when EI is low
    scores = ei * cost_penalty + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
