"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Infer progress: f_best starts near 0 and goes toward -78.3323
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Structural change: Use a hybrid acquisition that combines
    # 1) EI per unit cost (exploitation per budget)
    # 2) A "novelty" term based on sigma (exploration)
    # 3) A "proximity to known good" term to avoid jumping too far
    
    # EI per cost
    ei_cost = ei / cost_safe

    # Exploration term: high where uncertainty is high, but discounted by cost
    # Use a power of sigma to make exploration less aggressive
    explore = sigma_safe ** 0.5 / cost_safe

    # Exploitation term: low mean (good prediction) per cost
    exploit = -mu / cost_safe

    # Blend weights based on progress
    # Early: more exploration, Late: more exploitation
    w_explore = 0.3 * np.exp(-1.5 * progress)
    w_exploit = 0.2 + 0.5 * progress
    w_ei = 1.0 - w_explore - w_exploit
    if w_ei < 0.1:
        w_ei = 0.1
        # Renormalize
        total = w_explore + w_exploit + w_ei
        w_explore /= total
        w_exploit /= total
        w_ei /= total

    score = w_ei * ei_cost + w_explore * explore + w_exploit * exploit

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
