"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track when we started to avoid exceeding time limit
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the deadline
    elapsed = time.time() - _start_time
    if elapsed > 290:  # 10 seconds before the 300s limit
        return np.ones(2000) / cost
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Expected improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -30, 30)  # Clip to avoid numerical issues
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus: prefer points with higher uncertainty
    # This helps escape local optima
    exploration = sigma * 0.1
    
    # Cost-aware: divide by cost to favor cheaper points
    # Use a slight boost for points with very low cost
    cost_factor = 1.0 / (cost + 1e-9)
    
    # Combine: expected improvement per unit cost + exploration
    score = (ei + exploration) * cost_factor
    
    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 1e-12)
    
    return score
# EVOLVE-BLOCK-END
