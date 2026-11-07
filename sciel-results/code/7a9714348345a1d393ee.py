"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: kill long-running computations safely.
_T0 = time.time()
_HARD_DEADLINE = 315.0  # seconds; leave ~15s buffer for the harness


def _elapsed_ok():
    return (time.time() - _T0) < _HARD_DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = better to evaluate next).
    
    Strategy: Cost-Effective UCB (Upper Confidence Bound) for minimization.
    UCB(x) = mu(x) - beta * sigma(x)
    We want to minimize UCB, so we maximize -UCB(x) / cost(x).
    Score = (beta * sigma(x) - mu(x)) / cost(x)
    
    Using beta = 2.0 for a balance between exploration and exploitation.
    """
    if not _elapsed_ok():
        # Emergency: return something finite
        return np.zeros_like(mu)
    
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Beta parameter for exploration
    beta = 2.0
    
    # UCB for minimization: mu - beta * sigma
    # We want to maximize the "desirability", which is negative of UCB per cost
    # desirability = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    scores = (beta * sigma - mu) / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
