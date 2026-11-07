"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Griewank-2D.
    Combines Cost-Effective Expected Improvement (CEI) with Adaptive UCB.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct and inputs are safe
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: fall back to a fast single-variant score (CEI)
    if time.time() - _START > _LIMIT:
        z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
        s = ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
        return np.where(np.isfinite(s), s, 0.0)

    # 1. Expected Improvement per unit cost (CEI)
    z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    scores = ei / cost

    # 2. Adaptive UCB
    # Exploration strength decreases as f_best approaches the global minimum (0).
    # f_best for Griewank is in [0, 1]. 
    # If f_best is large, we are far from the minimum, explore more.
    # If f_best is small, we are close, exploit more.
    ef = float(np.clip(f_best / 1.5, 0.0, 1.0))
    kappa_base = 0.5 + 2.0 * ef
    
    # Portfolio of UCB variants; element-wise max implements the search.
    # We try a range of multipliers to balance exploration/exploitation.
    for mult in (0.5, 1.0, 1.5, 2.0):
        k = kappa_base * mult
        ucb = (f_best - mu + k * sigma) / cost
        scores = np.maximum(scores, ucb)

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
