# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 315s as safety margin
_DEADLINE = 315.0
_START_TIME = time.time()


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski-Tang 2D.

    Parameters
    ----------
    mu : array, shape (2000,)
        GP mean predictions in objective units.
    sigma : array, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : array, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled_to_01).

    Returns
    -------
    scores : array, shape (2000,)
        Acquisition scores (higher = more promising to evaluate).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)

    # Deadline guard: return a safe finite fallback
    if _check_deadline():
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # --- Core EI for minimization ---
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)

    # --- Adaptive UCB with a smoother, more informed schedule ---
    # Estimate total budget as roughly 30.
    # Use median cost as a robust proxy for the current "center" of the search.
    median_cost = float(np.median(cost_safe))
    # progress in [0,1]; early on costs are near 1, later the mean cost will be higher
    # as the GP focuses on more expensive (but potentially better) regions.
    progress = np.clip(median_cost / 30.0, 0.0, 1.0)

    # kappa: start high (exploration), decay toward exploitation
    # Use a smoother decay: kappa = kappa_min + (kappa_max - kappa_min) * (1 - progress)^1.5
    kappa_max = 2.5
    kappa_min = 0.4
    kappa = kappa_min + (kappa_max - kappa_min) * (1.0 - progress) ** 1.5

    ucb = -mu + kappa * sigma_safe

    # --- Normalize and blend ---
    diff = np.maximum(f_best - mu, 0.0)
    ref_ei = max(float(np.mean(diff)), 1e-6)
    ref_ucb = max(float(np.std(mu)), 1e-6)

    ei_norm = ei / ref_ei
    ucb_norm = ucb / ref_ucb

    # Blend weights: shift from UCB to EI as we progress
    # Early: 70% UCB, 30% EI; Late: 20% UCB, 80% EI
    w_ei = 0.25 + 0.65 * progress ** 1.2
    w_ucb = 1.0 - w_ei

    blended = w_ucb * ucb_norm + w_ei * ei_norm

    # --- Cost-aware scaling ---
    # cost in [1, 2]; we want to prefer cheaper points, but not overwhelmingly.
    # Use a softer penalty: divide by cost^alpha with alpha ~ 0.4
    cost_factor = cost_safe ** 0.4
    score = blended / cost_factor

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
