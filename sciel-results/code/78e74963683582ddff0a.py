"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() >= _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return a safe default if we're running out of time
    if _check_deadline():
        # Return cost-based scores (prefer low cost) as a safe fallback
        return 1.0 / (cost + 1e-9)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but clip for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost^2
    # This strongly penalizes high-cost evaluations, biasing search
    # toward the central region where the moved minimum is more likely to be
    score = ei / (cost * cost + 1e-12)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
