"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_EPS = 1e-9
_START_TIME = None
_DEADLINE = 280.0  # safety margin before 330s kill


def _ensure_deadline():
    """Return True if we should continue, False if we've hit the safety deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < _DEADLINE


def _compute_pi(mu, sigma, f_best):
    """Compute Probability of Improvement."""
    z = (f_best - mu) / (sigma + _EPS)
    z_clipped = np.clip(z, -10, 10)
    pi = norm.cdf(z_clipped)
    return np.maximum(pi, 0.0)


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Probability of Improvement with exponential cost discounting.
    Maximizes PI * exp(-lambda * cost), which directly balances improvement probability
    against evaluation cost in a principled multiplicative way.
    
    Args:
        mu: numpy array of shape (2000,), predicted means
        sigma: numpy array of shape (2000,), predicted standard deviations
        f_best: float, best observed value so far
        cost: numpy array of shape (2000,), evaluation costs for each candidate
        
    Returns:
        numpy array of shape (2000,), acquisition scores (higher is better)
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Convert to numpy arrays for consistent handling
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma = np.maximum(sigma, _EPS)
    cost = np.maximum(cost, _EPS)
    
    # Compute Probability of Improvement
    pi = _compute_pi(mu, sigma, f_best)
    
    # Exponential cost discounting: higher cost -> exponentially lower score
    # lambda=2.0 gives strong preference for cheap evaluations
    lam = 2.0
    cost_discount = np.exp(-lam * cost)
    
    # Final score: PI * exp(-lambda * cost)
    scores = pi * cost_discount
    
    # Check deadline
    if not _ensure_deadline():
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
