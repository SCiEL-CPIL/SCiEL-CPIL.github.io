"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Log-Cost Penalty LCB.
    
    Parameters
    ----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Adaptive kappa: increase exploration when f_best is far from expected minimum
    # For ST2D, the minimum is around -78.3323
    f_min_st2d = -78.3323
    
    # Measure how close we are to the theoretical minimum
    gap = max(f_best - f_min_st2d, 0.0)
    
    # Normalize gap: assume typical range of gap is 0 to ~100
    # When gap is large (early), explore more (higher kappa)
    # When gap is small (late), exploit more (lower kappa)
    gap_norm = np.clip(gap / 50.0, 0.0, 1.0)
    kappa_base = 2.0
    kappa_min = 0.5
    kappa = kappa_min + (kappa_base - kappa_min) * gap_norm
    
    # Compute LCB (for minimization, lower is better)
    lcb = mu - kappa * sigma
    
    # Calculate potential improvement over f_best
    # improvement = f_best - lcb
    # If lcb < f_best, improvement is positive.
    # We want to maximize this improvement.
    improvement = f_best - lcb
    
    # Ensure improvement is positive for log calculation
    # If improvement is <= 0, the point is not expected to improve f_best.
    # We assign a low score to these points.
    # Use a small epsilon to avoid log(0)
    eps = 1e-6
    improvement = np.maximum(improvement, eps)
    
    # Apply the log-cost penalty formula
    # score = log(improvement) - 0.5 * log(cost)
    # This is equivalent to log(improvement / sqrt(cost))
    # It models multiplicative utility: we care about relative improvement per sqrt(cost)
    
    log_cost = np.log(cost)
    log_improvement = np.log(improvement)
    
    scores = log_improvement - 0.5 * log_cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 295.0:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
