# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses cost-adjusted Expected Improvement with exploration bonus.
    Score = EI/scale - penalty * cost + bonus * sigma / scale
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI relative to problem scale for better conditioning
    scale = max(abs(f_best), np.mean(np.maximum(ei, 1e-10)), 1e-6)
    ei_scaled = ei / scale
    
    # Cost penalty: linear subtraction rather than division
    # This allows high-EI points in expensive regions to still be selected
    # if their improvement outweighs the cost
    cost_penalty = 0.3 * cost_safe
    
    # Exploration bonus: encourages visiting high-uncertainty regions
    # Scaled relative to the improvement scale
    exploration_bonus = 0.25 * sigma_safe / scale
    
    # Combined score: higher is better
    scores = ei_scaled - cost_penalty + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
