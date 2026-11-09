import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 285

def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT

def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    n = mu.size

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Ensure cost is positive and finite
    # Cost is 1 + mean(x_scaled), so min cost is 1.0 (if x_scaled=0)
    # Wait, x in [-5, 5] -> x_scaled in [0, 1].
    # If x = -5, x_scaled = 0. Cost = 1.
    # If x = 5, x_scaled = 1. Cost = 2.
    # So cost is in [1, 2].
    cost = np.maximum(cost, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is always non-negative for minimization
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-Aware Score: EI / Cost
    # Structural change: Linear cost penalty instead of square-root.
    # This creates a "global efficiency" metric (improvement per unit cost)
    # which is more aggressive in penalizing expensive points, potentially
    # focusing the search on cheaper regions or requiring higher confidence
    # for expensive points.
    score = ei / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Fallback for near-timeout
    if not _elapsed_ok():
        # Simple EI/cost as a fast fallback (same as main logic now)
        fallback = score
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    return score
