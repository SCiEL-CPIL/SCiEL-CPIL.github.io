"""Rewritten program: cost-aware acquisition for BO on shifted Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # safety margin before 300s / 330s kill


def _is_deadline_exceeded():
    return time.time() >= _DEADLINE


def _expected_improvement(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    return np.where(np.isfinite(ei), ei, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB.

    Parameters
    ----------
    mu : (N,) array, GP mean in objective units
    sigma : (N,) array, GP std in objective units
    f_best : float, best observed value (lowest)
    cost : (N,) array, cost of evaluating each candidate

    Returns
    -------
    (N,) array of finite scores; higher = evaluate next.
    """
    if _is_deadline_exceeded():
        # Fallback: rank by low mu / high uncertainty
        return np.where(np.isfinite(mu), -mu, -1e6)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # UCB for minimization: lower is better
    # UCB = -mu + beta * sigma (we want to maximize this score)
    # Adaptive beta based on number of evaluations implied by total cost
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5, so ~8-10 evals possible
    # Use a moderate exploration coefficient
    beta = 1.5

    # UCB score (higher = more promising for minimization)
    ucb_score = -mu + beta * sigma_safe

    # Cost-aware: divide by cost
    ucb_cost_aware = ucb_score / cost_safe

    # Add a small EI component for exploitation
    ei = _expected_improvement(mu, sigma, f_best)
    ei_cost_aware = ei / cost_safe

    # Blend: primarily UCB with some EI
    # This structural change increases exploration to escape local minima
    score = 0.7 * ucb_cost_aware + 0.3 * ei_cost_aware

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
