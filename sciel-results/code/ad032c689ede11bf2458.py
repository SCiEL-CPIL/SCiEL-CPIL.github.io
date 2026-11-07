"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 330.0  # Wall-clock limit for the evaluator
_SAFETY_MARGIN = 5.0  # Seconds to leave before the limit


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the time limit, return a simple fallback
    if _elapsed() > _TIME_LIMIT - _SAFETY_MARGIN:
        # Return a simple score based on sigma (explore) to stay valid
        return sigma.copy()
    
    # Cost-aware expected improvement with small exploration bonus
    # Use a small epsilon to avoid division by zero
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization
    cost_safe = np.maximum(cost, eps)
    score = ei / cost_safe
    
    # Add a small exploration term to avoid premature convergence
    # This encourages evaluating points with high uncertainty
    exploration_weight = 0.01
    score += exploration_weight * sigma / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
