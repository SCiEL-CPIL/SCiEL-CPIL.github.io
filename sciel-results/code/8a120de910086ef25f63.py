"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 10.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _start_time) - _MARGIN

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by cost^alpha.
    
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    This formulation captures the magnitude of expected improvement,
    which is often more informative than the probability of improvement (PI).
    Dividing by cost^alpha makes it cost-aware.
    """
    # Deadline guard
    if _remaining_time() < 1.0:
        # Fast fallback: simple PI / cost
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        cost_safe = np.maximum(cost, 1e-8)
        alpha = 1.5
        scores = pi / (cost_safe ** alpha)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost safety
    cost_safe = np.maximum(cost, 1e-8)
    
    # Cost-aware: divide by cost^alpha
    # Using alpha=1.0 balances improvement magnitude against cost efficiency.
    alpha = 1.0
    scores = ei / (cost_safe ** alpha)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
