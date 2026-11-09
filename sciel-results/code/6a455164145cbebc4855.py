"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    # Cost-aware EI: use EI with a small penalty for high cost
    # The cost structure is 1 + mean(x_scaled), so cost ranges [1, 2]
    # We want to prefer points with lower cost when EI is similar
    
    # Standard EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization: cost is in [1, 2], so 1/cost is in [0.5, 1]
    # This naturally downweights expensive points
    cost_inv = 1.0 / np.maximum(cost, 1e-9)
    
    # Combine EI with cost awareness
    # Use a weighted combination that emphasizes exploration in low-cost regions
    # The weight on cost should be small enough to not dominate EI
    cost_weight = 0.1
    score = ei * cost_inv - cost_weight * cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
