import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: use a non-linear cost penalty
    # Normalize cost to [0, 1] range relative to the current batch
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
    
    # Non-linear cost penalty: higher cost gets penalized more quadratically
    # This makes the acquisition more sensitive to cost differences
    cost_penalty = 1.0 / (1.0 + 2.0 * cost_norm ** 1.5)
    
    # Adaptive exploration: use the fraction of candidates with high uncertainty
    # to modulate exploration vs exploitation
    # High uncertainty regions get a boost to encourage exploration
    uncertainty_rank = np.argsort(np.argsort(sigma_safe)) / len(sigma_safe)
    # Exploration bonus: higher for candidates with high sigma
    exploration_bonus = 0.3 * uncertainty_rank
    
    # Combine: EI with cost penalty and exploration bonus
    score = ei * cost_penalty * (1.0 + exploration_bonus)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
