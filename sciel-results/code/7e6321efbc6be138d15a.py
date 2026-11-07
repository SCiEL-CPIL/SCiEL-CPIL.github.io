"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Probability of Improvement (PI) for
    Styblinski-Tang 1D on [-5, 5].
    
    PI directly maximizes P(f(x) < f_best | X=x), which focuses the search
    on the most promising basins identified by the GP posterior. This
    structural approach shifts from heuristic exploration-exploitation
    balancing to direct probabilistic improvement, which can break
    stagnation in local minima.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety: return PI score to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        return np.where(np.isfinite(pi), pi, 0.0)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for PI: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Add a small Expected Improvement component to break ties among
    # points with similar PI but different expected gains
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Blend: primarily PI, with small EI tie-breaker
    # PI values are in [0, 1], EI_norm is in [0, 1]
    # Weight: 0.95 PI + 0.05 EI to keep PI dominant
    score = 0.95 * pi + 0.05 * ei_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
