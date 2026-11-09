# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    UCB-based acquisition with adaptive exploration and cost-awareness.
    
    Uses: score = (f_best - mu) + beta * sigma - gamma * cost_normalized
    where beta adapts to exploration phase, and cost normalization
    penalizes expensive evaluations more heavily.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Estimate how many evaluations have been done from cost
    # Initial 5 points cost ~5 * (1 + 0.5) = 7.5 on average
    # Each subsequent eval costs 1 + mean(x in [0,1]) so avg ~1.5
    # Total cost budget is 30, so roughly (30 - 7.5) / 1.5 ≈ 15 more evals
    # We can estimate eval_count from total_cost if we track it, but we don't have it directly
    # Instead, use the magnitude of f_best as a proxy: lower f_best means more progress
    # Or use the spread of sigma: as GP matures, sigma decreases
    
    # Adaptive beta: start high for exploration, decrease as we gain confidence
    # Use the mean sigma as a proxy for GP maturity
    mean_sigma = np.mean(sigma_safe)
    # When mean_sigma is large, GP is uncertain -> explore more
    # When mean_sigma is small, GP is confident -> exploit more
    # Map mean_sigma from [0, 1] range to beta in [0.5, 3.0]
    beta = 0.5 + 2.5 * np.clip(mean_sigma, 0, 1)
    
    # UCB score: favor low mu (good predictions) and high sigma (uncertainty)
    # Since we're minimizing, low mu is good, so (f_best - mu) is positive when mu < f_best
    ucb_score = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware penalty: penalize expensive evaluations
    # Normalize cost to [0, 1] range for this batch
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost_safe - cost_min) / cost_range
    
    # Apply cost penalty: expensive points get lower scores
    # The penalty scales with how far above the minimum cost this point is
    gamma = 0.3  # Cost penalty weight
    cost_penalty = gamma * cost_norm
    
    # Final score: UCB minus cost penalty
    score = ucb_score - cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
