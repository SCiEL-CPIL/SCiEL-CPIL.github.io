"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    
    mu: shape (2000,) - GP mean
    sigma: shape (2000,) - GP std
    f_best: float - best observed value
    cost: shape (2000,) - evaluation costs
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a random or neutral score if we are running out of time
        # This prevents crashing the whole run, though it shouldn't happen
        return np.random.rand(len(mu))
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
