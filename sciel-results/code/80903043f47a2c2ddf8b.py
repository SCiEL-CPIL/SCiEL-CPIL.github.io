"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s. We reserve a safety margin.
_DEADLINE = 330.0
_START_TIME = time.time()


def _check_deadline():
    """Return True if we are approaching the deadline (safety margin 5s)."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a cost-normalized Lower Confidence Bound (LCB) with adaptive exploration
    based on the ratio of uncertainty to improvement potential.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: just use -mu / cost
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            scores = -mu / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and has a small floor
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute improvement potential: how much better could we do?
    # For minimization, improvement = f_best - mu (positive if mu < f_best)
    improvement = f_best - mu
    
    # Adaptive kappa: larger when we have high uncertainty relative to improvement
    # This balances exploration vs exploitation
    # When improvement is large (mu << f_best), we can be more exploitative (smaller kappa)
    # When improvement is small or negative, we explore more (larger kappa)
    
    # Base LCB: mu - kappa * sigma
    # We want to find points where the lower bound is most negative per unit cost
    
    # Adaptive kappa based on the spread of improvements
    # Use a simple heuristic: kappa scales with the ratio of sigma to typical improvement
    typical_improvement = np.max(np.abs(improvement)) + 1e-9
    typical_sigma = np.mean(sigma) + 1e-9
    
    # kappa in range [0.5, 3.0]
    # Higher when uncertainty dominates
    kappa = 1.5 + 1.5 * (typical_sigma / (typical_improvement + typical_sigma))
    kappa = np.clip(kappa, 0.5, 3.0)
    
    # LCB score: lower is better, so we negate for "higher is better" scoring
    lcb = mu - kappa * sigma
    
    # Cost-normalized: divide by cost to account for evaluation expense
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = -lcb / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
