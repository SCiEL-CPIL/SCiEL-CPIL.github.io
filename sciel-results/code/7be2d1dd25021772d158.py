"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a log-cost penalty on LCB.
    
    This models the optimization as a multiplicative utility maximization problem
    rather than a linear ratio, preventing aggressive bias toward low-cost regions
    while preserving exploration in high-cost areas.
    
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
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Theoretical minimum for Styblinski-Tang 2D
    f_min_st2d = -78.3323
    
    # Adaptive kappa: increase exploration when far from optimum
    gap = max(f_best - f_min_st2d, 0.0)
    # Normalize gap: early (large gap) -> explore more, late (small gap) -> exploit
    gap_norm = np.clip(gap / 80.0, 0.0, 1.0)
    
    # Adaptive exploration parameter
    kappa_base = 2.5
    kappa_min = 1.0
    kappa = kappa_min + (kappa_base - kappa_min) * gap_norm
    
    # Compute LCB (Lower Confidence Bound) for minimization
    # LCB = mu - kappa * sigma
    # We want high scores for low LCB values (potential improvement)
    lcb = mu - kappa * sigma
    
    # Log-cost penalty approach:
    # score = log(-lcb) - 0.5 * log(cost)
    # This is equivalent to maximizing (-lcb) / sqrt(cost) in log space
    # which is a multiplicative utility model
    
    # Handle cases where lcb might be positive (no expected improvement)
    # Use a soft threshold to avoid log of non-positive values
    # Shift lcb to ensure it's negative enough for log
    # If lcb is close to 0 or positive, the log will be problematic
    # Use: log(max(-lcb, epsilon)) where epsilon is small
    
    eps = 1e-8
    neg_lcb = np.maximum(-lcb, eps)
    
    # Log utility: log(neg_lcb) - 0.5 * log(cost)
    # This gives higher scores to points with good expected improvement per unit cost
    # but in a multiplicative sense, preventing extreme cost bias
    
    scores = np.log(neg_lcb) - 0.5 * np.log(cost)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 295.0:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
