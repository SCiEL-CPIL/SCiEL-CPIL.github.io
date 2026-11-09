"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    for minimization, with cost normalization.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        # Simple greedy fallback: minimize mu per unit cost
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware EI: divide by cost to account for evaluation expense
    # This naturally balances exploration (high sigma → high EI) 
    # and exploitation (low mu → high EI) while penalizing expensive points
    score = ei / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
