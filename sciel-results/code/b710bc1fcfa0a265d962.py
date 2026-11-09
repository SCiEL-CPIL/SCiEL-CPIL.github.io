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
    Cost-Aware UCB (Upper Confidence Bound).
    
    Unlike EI which estimates expected gain, UCB optimizes for the
    upper bound of potential improvement, encouraging exploration
    in high-uncertainty regions. The cost is handled by normalizing
    the UCB score, penalizing expensive points.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Structural Change: Switch from Expected Improvement (EI) to
    # Upper Confidence Bound (UCB).
    # UCB = mu + beta * sigma
    # We use a fixed beta that balances exploration/exploitation.
    # For standardized GP outputs, beta ~ 2.0-3.0 is typical.
    # We adjust beta slightly based on the overall uncertainty to 
    # ensure robustness, but keep it simple (no complex adaptive weights).
    
    # A simple, robust UCB score
    beta = 2.5 
    ucb = mu + beta * sigma
    
    # Cost-Aware Normalization
    # We divide by cost to prefer cheaper points that offer similar UCB potential.
    # To avoid division by zero or extreme values, we use cost_safe.
    score = ucb / cost_safe

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
