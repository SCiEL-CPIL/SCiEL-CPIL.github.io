"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 40s margin before 330s kill


def _time_guard():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines Expected Improvement with budget-scaled exploration.
    Returns 2000 finite scores; higher = evaluate next.
    """
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Track number of acquisition calls to estimate budget usage
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    n_calls = acquisition._call_count

    # Estimate total cost used so far
    # Initial 5 points + n_calls evaluations
    # Average cost ~1.5 (since x is uniform on [0,1] scaled, mean=0.5, cost=1.5)
    # But we can be more precise: cost = 1 + mean(x_scaled)
    # For uniform on [0,1], mean(x_scaled) = 0.5, so avg cost = 1.5
    # Total cost ≈ 5 * 1.5 + n_calls * 1.5 = (5 + n_calls) * 1.5
    total_cost_est = (5.0 + n_calls) * 1.5

    # Budget is 30.0
    budget = 30.0
    # Remaining budget fraction: 1 at start, 0 at end
    remaining_frac = np.clip(1.0 - total_cost_est / budget, 0.0, 1.0)

    # Exploration coefficient: high early, low late
    # kappa = kappa_min + (kappa_max - kappa_min) * remaining_frac
    kappa_max = 2.0
    kappa_min = 0.3
    kappa = kappa_min + (kappa_max - kappa_min) * remaining_frac

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Exploration term: reward uncertainty
    # Use a scaled version of sigma
    exploration = kappa * s

    # Combine: weighted sum of EI and exploration
    # Weight EI more heavily as we have more data
    # weight_ei increases with n_calls
    weight_ei = np.clip(0.3 + 0.1 * n_calls, 0.3, 1.5)
    weight_explore = 1.0

    score = weight_ei * ei + weight_explore * exploration

    # Cost-aware: divide by cost to favor cheap evaluations
    score = score / c

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
