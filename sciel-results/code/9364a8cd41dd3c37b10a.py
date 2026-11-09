"""Cost-aware acquisition for Hartmann 6d Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = 315.0
_start_time = time.time()


def _check_time():
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a cost-normalized Expected Improvement (EI) strategy.
    EI for minimization: E[max(f_best - f(x), 0)]
    We maximize the score: EI / cost
    """
    if _check_time():
        return np.ones(2000)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x) if f(x) < f_best, else 0.
    # f(x) ~ N(mu, sigma^2)
    # Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Clip z to avoid numerical issues in extreme tails, though norm handles it reasonably
    # but for very large |z|, phi(z) is 0 and Phi(z) is 0 or 1.
    # If z is very negative (mu >> f_best), EI should be ~0.
    # If z is very positive (mu << f_best), EI should be ~ f_best - mu.
    
    # Using scipy.stats.norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = delta * Phi + sigma_safe * phi
    
    # The score is EI / cost
    # This encourages regions with high expected improvement relative to cost
    scores = ei / cost_safe
    
    # Handle NaN/inf
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
