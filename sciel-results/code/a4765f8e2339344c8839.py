"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use a 325s deadline to be safe.
_start_time = time.time()
_DEADLINE = 325.0


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a simple score
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return (f_best - mu) / cost_safe

    # Cost-aware Expected Improvement with exploration bonus for Rastrigin's local minima
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # Standard EI for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Cost-aware: divide by cost
    score = ei / cost_safe

    # Add exploration bonus: prefer points with high uncertainty
    # This helps escape local optima in rastrigin's many local minima
    # Use a moderate weight to balance exploitation and exploration
    exploration_weight = 0.5
    score = score + exploration_weight * sigma_safe / cost_safe

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
