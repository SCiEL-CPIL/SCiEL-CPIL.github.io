"""Rewritten acquisition for Bayesian optimization of Branin 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function.
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: uniform scores (all equal)
        return np.ones_like(mu, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # POI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
