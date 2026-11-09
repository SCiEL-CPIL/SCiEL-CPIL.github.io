"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s. We reserve a safety margin.
_DEADLINE = 330.0
_START_TIME = time.time()


def _check_deadline():
    """Return True if we are approaching the deadline (safety margin 5s)."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Computes the expected improvement over f_best for each candidate,
    normalized by evaluation cost. This is a probabilistic acquisition
    that directly optimizes expected reduction per unit cost.
    
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
        # Fallback: simple cost-normalized negative mean
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
    
    # For minimization: improvement I = f_best - f(x)
    # E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10.0, 10.0)
    
    # Expected improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is always non-negative; points with mu > f_best get near-zero EI
    # Cost-normalized: expected improvement per unit cost
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
