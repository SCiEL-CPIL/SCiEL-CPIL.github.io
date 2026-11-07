"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are very close to the limit, use a simple exploitation heuristic
    # to ensure we finish the current iteration quickly and potentially avoid a long GP fit
    # in the next iteration (though we can't control that, it's a best-effort guard).
    if time.time() - _start_time > 300:
        # Return -mu / cost to favor low mu (good for minimization) and low cost
        # This is a simple exploitation strategy.
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Calculate z-score for Probability of Improvement
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Cost-aware score: Probability of Improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = pi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
