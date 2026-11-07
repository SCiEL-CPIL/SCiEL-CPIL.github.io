"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline enforcement
_PROGRAM_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    # 1. Deadline Check: return a fast, valid score if we're running out of time
    elapsed = time.time() - _PROGRAM_START_TIME
    if elapsed > _TIME_LIMIT:
        # Fast fallback: simple EI / cost
        safe_sigma = sigma + 1e-9
        z = (f_best - mu) / safe_sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # 2. Cost-aware Expected Improvement (EI)
    # EI is the standard choice for cost-aware Bayesian optimization.
    # We divide by cost to get "improvement per unit cost".
    
    eps = 1e-12
    safe_sigma = sigma + eps
    z = (f_best - mu) / safe_sigma
    
    # Standard EI formula for minimization:
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware score: EI / cost
    # This rewards points that have high expected improvement per unit cost.
    scores = ei / (cost + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
