"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        # Pure exploitation: prefer points with predicted mean below f_best,
        # normalized by cost.
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # --- Robust Expected Improvement (Student-t approximation) ---
    # We approximate the heavier tails of a Student-t distribution (nu=4)
    # by inflating the standard deviation. This makes the acquisition function
    # more robust to GP model misspecification, preventing premature convergence
    # to local minima where the GP is overconfident.
    # For nu=4, the inflation factor is sqrt(nu / (nu - 2)) = sqrt(4/2) = sqrt(2).
    inflation_factor = np.sqrt(2.0)
    sigma_robust = sigma_safe * inflation_factor
    
    # Standard EI calculation using the inflated sigma
    z = (f_best - mu) / sigma_robust
    # Clip z to avoid numerical issues in norm.cdf and norm.pdf
    z = np.clip(z, -10.0, 10.0)
    
    ei_robust = (f_best - mu) * norm.cdf(z) + sigma_robust * norm.pdf(z)
    
    # Handle potential numerical artifacts
    ei_robust = np.nan_to_num(ei_robust, nan=0.0, posinf=1e6, neginf=0.0)
    
    # --- Cost-aware scaling ---
    # cost is in [1, 2].
    # Dividing by cost penalizes expensive points.
    cost_safe = np.maximum(cost, 1.0)
    score = ei_robust / cost_safe
    
    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
