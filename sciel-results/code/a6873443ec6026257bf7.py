"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: total wall-clock budget is 330s, keep safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 320.0  # seconds, leave 10s margin before 330s kill


def _time_remaining():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: scalar, lowest observed value
    cost: (2000,) evaluation cost for each candidate
    
    Returns: (2000,) scores, higher = prefer to evaluate next
    """
    # Check time budget
    if _time_remaining() < 5.0:
        # Return a simple fallback: just -mu / cost to be safe
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (no improvement expected)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    scores = ei / (cost + 1e-9)
    
    # Add a small exploration term based on uncertainty to avoid
    # getting stuck in exploitation-only mode
    # Use a small weight so it doesn't dominate
    exploration_weight = 0.1
    # Normalize sigma for exploration term
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration = exploration_weight * sigma_safe / sigma_max
    
    scores = scores + exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
