"""Rewritten cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement divided by cost.
    EI naturally balances exploration and exploitation.
    Dividing by cost gives expected improvement per unit cost.
    """
    t0 = time.time()
    deadline = t0 + 290.0  # safety margin before 300s limit

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # z-score for improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware score: EI per unit cost
    score = ei / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    # Time guard: if we're close to deadline, return immediately
    if time.time() > deadline - 5.0:
        return score

    return score
# EVOLVE-BLOCK-END
