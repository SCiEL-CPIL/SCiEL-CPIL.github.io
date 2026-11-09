"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses Probability of Improvement (PI) divided by cost.
    PI is bounded in [0, 1] and provides a robust signal for minimization.
    Dividing by cost ensures that cheaper evaluations are preferred,
    which is crucial for cost-aware optimization.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / np.maximum(cost, 1e-9)
        return np.where(np.isfinite(scores), scores, 0.0)

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute PI: bounded in [0, 1], sigmoidal
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)

    # Cost-aware normalization
    scores = pi / cost_safe

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
