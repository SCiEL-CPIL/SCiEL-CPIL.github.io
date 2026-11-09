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
    cost = np.maximum(cost, 1.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # Structural Change: Use Upper Confidence Bound (UCB) instead of EI.
    # UCB balances exploitation (low mu) and exploration (high sigma).
    # For minimization, we maximize (beta * sigma - mu).
    # We use a cost-aware UCB: (beta * sigma - mu) / cost.
    
    # Dynamic beta: Use a value that scales with the uncertainty distribution.
    # A fixed beta is simpler and often robust. Let's try beta=1.0 first.
    # If sigma is in objective units and mu is in objective units,
    # beta=1.0 means we value 1 unit of uncertainty as much as 1 unit of mean reduction.
    
    beta = 1.0
    
    ucb = beta * sigma - mu
    
    # Ensure UCB is finite
    ucb = np.where(np.isfinite(ucb), ucb, -1e10)
    
    # Cost-aware score
    score = ucb / cost
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Fallback for near-timeout
    if not _elapsed_ok():
        # Simple UCB/cost as a fast fallback
        fallback = ucb / cost
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    return score
