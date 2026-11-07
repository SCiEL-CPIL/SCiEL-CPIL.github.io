"""Cost-aware acquisition for Levy 2D BO with variable evaluation cost."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_DEADLINE = time.time() + 300  # 300s wall-clock limit per evaluation

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition: EI per unit cost with exploration bonus."""
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    cost_safe = np.maximum(cost, 1e-6)
    ei_per_cost = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty relative to cost
    # This helps when EI is low but uncertainty is high in cheap regions
    exploration_bonus = sigma / cost_safe * 0.1
    
    # Combine
    score = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
