"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement per unit cost.
    Score = EI / cost.
    This directly optimizes the rate of improvement relative to the expense.
    """
    if time.time() - _START_TIME > _TIMEOUT:
        # Return a neutral score to allow the loop to finish/timeout safely
        return np.zeros(len(mu))

    n = len(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI / cost
    # This favors points that offer high expected improvement per unit cost
    score = ei / np.maximum(cost, 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
