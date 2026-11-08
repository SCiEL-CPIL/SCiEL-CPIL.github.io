"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Cache for the deadline
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _safe_norm_cdf(x):
    return norm.cdf(x)


def _safe_norm_pdf(x):
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value so far
        cost: (2000,) array of costs for each candidate
    
    Returns:
        (2000,) array of acquisition values (higher = more promising)
    """
    # Deadline guard: if we're getting close to the 330s limit, return a simple fallback
    if _get_elapsed() > 325:
        return np.full_like(mu, 0.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute expected improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * _safe_norm_cdf(z) + sigma_safe * _safe_norm_pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost (higher cost = lower score)
    # cost is in [1, 2] range (1 + mean(x scaled to [0,1]))
    cost_safe = np.maximum(cost, 1e-8)
    
    # Use a power of cost to make it more/less cost-sensitive
    # EI / cost is standard; let's try EI / (cost ** 1.5) to be more cost-sensitive
    scores = ei / (cost_safe ** 1.5)
    
    # Add a small exploration term to avoid always picking the same region
    # This encourages evaluating new points
    exploration = sigma_safe * 0.1
    
    scores = scores + exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
