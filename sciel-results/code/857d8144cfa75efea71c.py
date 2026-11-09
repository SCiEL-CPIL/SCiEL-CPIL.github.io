"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the time limit, return a simple exploitation score
    if time.time() > _DEADLINE:
        # Simple exploitation: prefer points with lowest mu
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Cost-aware Expected Improvement (EI)
    # EI balances exploration (sigma) and exploitation (mu) better than POI
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Standard EI formula: E[max(f_best - f, 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Divide by cost to account for evaluation cost
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    scores = ei / cost_safe
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
