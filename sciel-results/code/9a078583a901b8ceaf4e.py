"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 10s margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    Higher score means more preferred to evaluate next.
    Uses UCB (Upper Confidence Bound) divided by evaluation cost.
    """
    # Deadline guard: if we are close to the limit, return zeros (random selection)
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)

    # Use UCB (Upper Confidence Bound) for minimization
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so score = -LCB / cost = (beta * sigma - mu) / cost
    # Beta = 1.0 is a standard choice for exploration/exploitation balance
    beta = 1.0
    ucb_score = beta * sigma_safe - mu
    
    # Cost-aware: divide by cost
    scores = ucb_score / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
