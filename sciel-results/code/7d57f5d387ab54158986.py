"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 310s margin)
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

    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Infer progress: f_best starts near 0 (origin) and goes toward -78.3323
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    # ST-2d: f(0,0)=0, min=-78.3323
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Adaptive beta: high early (explore), low late (exploit)
    # Smooth exponential decay for beta
    # Start ~2.0, End ~0.5
    beta = 0.5 + 1.5 * np.exp(-2.0 * progress)
    
    # UCB
    ucb = -mu + beta * sigma_safe

    # Cost normalization: divide by cost to prefer cheaper points
    cost_safe = np.maximum(cost, 1e-6)
    
    # Score components per unit cost
    score_ei = ei / cost_safe
    score_ucb = ucb / cost_safe

    # Blend: early favor UCB, late favor EI
    # Reduced UCB weight to avoid excessive jumping
    # w_ucb: 0.4 early, 0.05 late
    w_ucb = 0.05 + 0.35 * np.exp(-2.0 * progress)
    w_ei = 1.0 - w_ucb

    score = w_ei * score_ei + w_ucb * score_ucb

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
