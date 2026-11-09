"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0  # seconds
_MARGIN = 5.0      # safety margin


def _elapsed():
    return time.time() - _start_time


def _deadline_exceeded():
    return _elapsed() > _DEADLINE - _MARGIN


def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if exceeded, return a deterministic fallback
    if _deadline_exceeded():
        # Return scores that prefer low cost and low mu as a safe fallback
        cost_safe = np.maximum(cost, 1e-9)
        fallback_score = -mu / cost_safe
        return np.nan_to_num(fallback_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Cost-normalized probability of improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Deterministic regret lower bound component:
    # Expected minimum value at x is approximately mu - sigma (optimistic)
    # So expected regret reduction is f_best - (mu - sigma) = f_best - mu + sigma
    # This is a linear, less exploitative alternative to EI's Gaussian terms
    regret_lower_bound = f_best - mu + sigma_safe
    
    # Combine: POI for probability of improvement + regret lower bound for magnitude
    # Both normalized by cost to be cost-aware
    score = (poi + np.maximum(regret_lower_bound, 0.0)) / cost_safe
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
