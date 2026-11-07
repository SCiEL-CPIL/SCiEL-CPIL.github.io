"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling for minimization.
    Returns 2000 finite scores; higher = evaluate next.
    
    Strategy:
    - Thompson Sampling: draw a sample from the GP posterior at each candidate
    - For minimization, we want the lowest sampled value
    - Score = -sampled_value / cost, so higher scores favor cheap, promising points
    - This naturally balances exploration (high sigma leads to wide sample distribution)
      and exploitation (low mu favors known good regions)
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: use mean-based score divided by cost
        c = np.maximum(cost, 1e-9)
        return -mu / c

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Thompson Sampling: sample from N(mu, sigma^2)
    # Use a single random draw per call (consistent across all 2000 candidates)
    # The sample is: mu + sigma * z where z ~ N(0, 1)
    z = np.random.standard_normal()
    
    # Sampled values
    sampled = mu + s * z
    
    # For minimization, we want the lowest sampled value
    # Score: -sampled_value / cost (higher is better)
    # This means: prefer low sampled values (good predictions) and low cost
    
    score = -sampled / c
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
