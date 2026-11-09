"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural shift:
    Pure cost-normalized Expected Improvement (EI) with adaptive
    temperature scaling based on uncertainty distribution.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Structural change: Instead of blending EI with UCB, use a
    # cost-aware EI with adaptive exploration via a temperature-like
    # scaling that amplifies exploration in early stages (high sigma
    # regions) and focuses on exploitation later.
    
    # Compute a normalized uncertainty measure to adapt behavior
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Adaptive scaling: when overall uncertainty is high, boost exploration
    # by scaling EI with a factor that depends on the relative uncertainty
    # of each candidate vs. the mean uncertainty
    relative_sigma = sigma / (sigma_mean + 1e-9)
    
    # Use a softplus-like activation to create adaptive weights
    # Candidates with higher relative uncertainty get a boost
    # This is a structural change: instead of fixed UCB blend, we use
    # uncertainty-adaptive EI scaling
    exploration_weight = 1.0 + 0.5 * np.log1p(relative_sigma)
    
    # Cost-normalized EI with adaptive exploration
    ei_score = ei * exploration_weight / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(ei_score), ei_score, 0.0)

    return score
# EVOLVE-BLOCK-END
