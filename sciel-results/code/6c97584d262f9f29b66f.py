"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """
    Expected Improvement: E[max(f_best - f(x), 0)]
    where f(x) ~ N(mu, sigma^2).
    """
    s = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    return np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    
    # Cost-aware scoring: EI / cost^0.8
    # Using cost^0.8 instead of cost^1 allows the acquisition function to select
    # points with higher potential improvement even if they are slightly more expensive,
    # potentially escaping local optima in the cost-aware selection while still respecting
    # the budget, which may lead to a better final f_best.
    ei = _expected_improvement(mu, sigma, f_best)
    
    c = np.maximum(cost, 1e-9)
    score = ei / (c ** 0.8)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
