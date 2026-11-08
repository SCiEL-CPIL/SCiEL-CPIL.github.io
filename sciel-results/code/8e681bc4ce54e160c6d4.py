"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE - 1.0:
        # Fallback: return random finite scores
        return np.random.randn(len(mu))
    
    # Use Expected Improvement (EI) instead of cost-aware EI
    # Cost is already baked into the search loop via the cost of evaluating
    # So we focus on pure EI to find the best point
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Handle cases where f_best - mu is very negative (mu >> f_best)
    # In those cases, EI is 0
    ei = np.zeros_like(mu)
    mask = (f_best - mu) > 0
    ei[mask] = (f_best[mask] - mu[mask]) * norm.cdf(z[mask]) + sigma_safe[mask] * norm.pdf(z[mask])
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
