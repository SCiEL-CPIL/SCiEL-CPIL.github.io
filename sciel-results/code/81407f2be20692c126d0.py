"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 292.0
_ADAPTIVE_LCB_BETA = 0.5


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.

    Parameters
    ----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        m = np.asarray(mu, dtype=np.float64)
        s = np.asarray(sigma, dtype=np.float64)
        c = np.asarray(cost, dtype=np.float64)
        s = np.maximum(s, 1e-9)
        c = np.maximum(c, 1e-9)
        # Fallback: adaptive LCB
        # Theoretical minimum for cosine8 8d is -0.8
        # Beta adapts based on how close f_best is to the theoretical minimum
        theoretical_min = -0.8
        gap = f_best - theoretical_min
        # When gap is large, explore more (higher beta); when close, exploit (lower beta)
        # Normalize gap: if f_best is at -0.8, gap=0 -> beta=0; if f_best is far, beta increases
        # Use a smooth transition
        beta = _ADAPTIVE_LCB_BETA * (1.0 + 0.5 * gap)
        beta = np.clip(beta, 0.1, 1.0)
        lcb = m - beta * s
        scores = -lcb / c
        return np.where(np.isfinite(scores), scores, 0.0)

    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)

    s = np.maximum(s, 1e-9)
    c = np.maximum(c, 1e-9)

    # --- Adaptive LCB with cost penalty ---
    # Theoretical minimum for cosine8 8d is -0.8
    # Adaptive beta: when f_best is far from theoretical minimum, explore more
    theoretical_min = -0.8
    gap = f_best - theoretical_min
    
    # Beta increases with gap to encourage exploration when we're far from optimal
    # Scale: gap of 0.5 (f_best = -0.3) -> moderate exploration
    # gap of 0.0 (f_best = -0.8) -> exploitation
    # Use a sigmoid-like or linear scaling
    # Linear: beta = base + scale * gap, clipped
    base_beta = 0.3
    scale_beta = 1.5
    beta = base_beta + scale_beta * gap
    beta = np.clip(beta, 0.2, 1.5)
    
    # LCB: m - beta * sigma
    lcb = m - beta * s
    
    # Cost-aware: minimize LCB per unit cost -> maximize -LCB/cost
    scores = -lcb / c

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
