"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) normalized by cost.
    """
    # Check deadline
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement for minimization
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-aware score: PI per unit cost
    # cost is in [1, 2]
    score = pi / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
