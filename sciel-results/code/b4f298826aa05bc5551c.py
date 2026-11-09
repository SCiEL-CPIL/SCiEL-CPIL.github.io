"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a multiplicative EI-UCB combination
    for minimization, with adaptive exploration.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    ei_cost = ei / cost_safe

    # UCB for minimization: higher score = more promising
    # ucb_value = mu - kappa * sigma (we want to minimize this)
    # So score contribution = -mu + kappa * sigma
    kappa = 1.5
    ucb_score = (kappa * sigma - mu) / cost_safe

    # Multiplicative combination: rewards points that are good on BOTH metrics
    # This avoids the "compromise" problem of linear blending
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    combined = np.log1p(ei_cost + eps) * np.log1p(np.maximum(ucb_score, 0.0) + eps)

    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in early iterations
    exploration_bonus = 0.1 * sigma / cost_safe

    score = combined + exploration_bonus

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
