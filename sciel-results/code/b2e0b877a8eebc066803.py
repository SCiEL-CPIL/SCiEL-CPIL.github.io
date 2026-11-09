"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time for safety.
_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0  # 15s safety margin


def _cost_aware_pi(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    PI = Phi((f_best - mu) / sigma) / cost
    
    This provides a bounded, sigmoidal exploration signal that:
    - Naturally saturates in high-uncertainty regions (avoids unbounded noise-chasing of UCB)
    - Offers less aggressive exploitation pressure than EI
    - Is normalized by cost to prefer cheaper evaluations
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    cost_safe = np.maximum(cost, 1e-9)
    scores = pi / cost_safe
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses cost-aware Probability of Improvement (PI):
    - Bounded in [0, 1] before cost normalization
    - Sigmoidal exploration that saturates in high-uncertainty regions
    - Naturally balances exploration and exploitation without tuning beta
    - Cost normalization prefers cheaper evaluations
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / np.maximum(cost, 1e-9)
        return scores

    # Compute cost-aware PI
    scores = _cost_aware_pi(mu, sigma, f_best, cost)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
