"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline


def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a hybrid strategy combining:
    1. Expected Improvement (EI) for exploitation
    2. Lower Confidence Bound (LCB) for exploration
    
    The blend weight adapts based on total cost (progress), starting with
    more exploration and shifting to exploitation as budget is consumed.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at 2000 candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at 2000 candidate points.
    f_best : float
        Lowest observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of coordinates scaled to [0,1]).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    if _check_timeout():
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle edge cases: non-positive sigma
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)

    # Compute Expected Improvement (EI)
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute Lower Confidence Bound (LCB) score
    # Adaptive exploration coefficient: starts high, decays with cost
    # Cost ranges from ~5 (initial) to 30
    # At cost=5, beta=2.5 (high exploration). At cost=30, beta=0.5 (exploitation).
    avg_cost = np.mean(cost)
    beta = 2.5 - 0.1 * avg_cost
    beta = np.clip(beta, 0.5, 3.0)

    # LCB potential improvement: how much better could the optimistic bound be than f_best?
    lcb = mu - beta * sigma_safe
    lcb_improvement = np.maximum(0.0, f_best - lcb)

    # Cost-normalized scores
    ei_score = ei / cost_safe
    lcb_score = lcb_improvement / cost_safe

    # Blend weight: more exploration early, more exploitation later
    # progress = (avg_cost - 5) / (30 - 5), clamped to [0, 1]
    progress = (avg_cost - 5.0) / 25.0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Weight for EI (exploitation) increases with progress
    # Weight for LCB (exploration) decreases with progress
    # Use a smooth transition: w_ei = 0.3 + 0.7 * progress, w_lcb = 1 - w_ei
    w_ei = 0.3 + 0.7 * progress
    w_lcb = 1.0 - w_ei

    # Hybrid score
    scores = w_ei * ei_score + w_lcb * lcb_score

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
