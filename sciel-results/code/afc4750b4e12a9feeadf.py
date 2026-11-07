import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 315s margin)
_T0 = time.time()
_DEADLINE = 315.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    Combines LCB-based exploitation with EI for improvement estimation,
    and uses progress-adaptive exploration.
    
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
        beta = 1.0
        lcb = mu - beta * sigma_safe
        score = -lcb / np.maximum(cost, 1e-6)
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

    # Infer progress: f_best starts near 0 (origin) and goes toward -78.3323
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    # ST-2d: f(0,0)=0, min=-78.3323
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive beta: start with more exploration, reduce as we converge
    # Early: beta ~ 2.5, Late: beta ~ 0.5
    beta = 0.5 + 2.0 * (1.0 - progress) ** 1.5

    # Lower Confidence Bound (potential minimum)
    lcb = mu - beta * sigma_safe

    # Combine EI and LCB with adaptive weights
    # Early: more weight on LCB (exploration), Late: more weight on EI (exploitation)
    w_lcb = 0.3 + 0.4 * (1.0 - progress)  # 0.7 early, 0.3 late
    w_ei = 1.0 - w_lcb                     # 0.3 early, 0.7 late

    # Score: higher is better
    # -lcb is high when lcb is low (good minimum)
    # ei is high when improvement is likely
    score = (w_lcb * (-lcb) + w_ei * ei) / cost_safe

    # Ensure output is finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
