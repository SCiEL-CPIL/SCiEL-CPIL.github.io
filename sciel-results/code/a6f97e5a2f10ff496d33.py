"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track start time for the acquisition function
_acq_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Safety margin: 320s limit (10s before the 330s kill)
    elapsed = time.time() - _acq_start_time
    if elapsed > 320:
        # Fallback: return EI per cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost
    ei_cost = ei / (cost + 1e-9)
    
    # Diversity-promoting penalty: penalize points with low predictive variance
    # This encourages exploration of uncertain regions
    # Use a soft penalty based on sigma relative to its max
    sigma_max = np.max(sigma) + 1e-9
    diversity_bonus = sigma / sigma_max  # Higher sigma = more uncertain = more diverse
    
    # Combine: weighted sum of EI/cost and diversity
    # The diversity term helps avoid clustering around already-evaluated points
    # since low-sigma regions have been well-explored
    weight_diversity = 0.1
    
    scores = ei_cost * (1 + weight_diversity * diversity_bonus)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
