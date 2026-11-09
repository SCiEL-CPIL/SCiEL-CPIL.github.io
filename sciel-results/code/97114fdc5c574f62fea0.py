"""Cost-aware acquisition using Expected Regret per unit cost for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a safety margin below the 300 s wall-clock limit.
_START_TIME = time.time()
_DEADLINE = 285.0  # seconds


def _check_deadline():
    """Return True if we are close to the wall-clock deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def _expected_regret(mu, sigma, f_best):
    """
    Expected Regret: E[max(f_best - f(x), 0)] for a Gaussian f(x) ~ N(mu, sigma^2).
    This is the same as EI but we use it directly as the objective to maximize per cost.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # ER = (f_best - mu) * Phi(z) + sigma * phi(z)
    er = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure non-negative
    er = np.maximum(er, 0.0)
    return er


def _probability_of_improvement(mu, sigma, f_best):
    """Probability that f(x) < f_best."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Uses Expected Regret (ER) normalized by cost, with an adaptive exploration
    component that fades as progress is made.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: if we are near the wall-clock limit, return a safe
    # exploitation-focused score immediately.
    if _check_deadline():
        cost_safe = np.maximum(cost, 1e-9)
        fallback = -mu / cost_safe
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback

    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Regret (same as EI for minimization)
    er = _expected_regret(mu, sigma, f_best)

    # Compute probability of improvement as a secondary exploration signal
    pi = _probability_of_improvement(mu, sigma, f_best)

    # Progress metric: 0 when f_best is large (early), 1 when f_best ~ 0 (late)
    # Rastrigin 2d max value is around 40, so normalize by that
    progress = float(np.clip(1.0 - f_best / 40.0, 0.0, 1.0))

    # Adaptive exploration weight: high early, low late
    # The exploration term encourages sampling in uncertain regions
    w_explore = 0.5 * (1.0 - progress)  # 0.5 -> 0.0

    # Exploration component: use sigma directly as a measure of uncertainty,
    # weighted by PI to only explore where improvement is plausible
    exploration = w_explore * sigma * pi

    # Combine ER and exploration
    combined = er + exploration

    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
