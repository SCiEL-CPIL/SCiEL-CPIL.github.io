"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def _safe_divide(a, b, eps=1e-9):
    """Safe element-wise division with epsilon guard."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(np.abs(b) > eps))

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Probability of Improvement (PI) divided by the cost.
    PI = Phi((f_best - mu) / sigma), which is bounded in [0, 1] and provides
    a smooth, sigmoidal exploration signal. Dividing by cost (not sqrt) makes
    the penalty stronger, discouraging expensive evaluations more aggressively
    while still allowing the GP to guide the search towards promising regions.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    diff = f_best - mu
    z = _safe_divide(diff, sigma)
    
    # Clip z to avoid numerical issues in norm.cdf for very large values
    z = np.clip(z, -10, 10)
    
    pi = norm.cdf(z)
    
    # Ensure non-negative PI (numerical safety)
    pi = np.maximum(pi, 0.0)
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    # Cost is in [1, 2]. Dividing by cost directly makes the acquisition
    # function more sensitive to cost differences, which can help avoid
    # the corners where cost is lowest but the minimum is unlikely to be.
    cost_safe = np.maximum(cost, 1e-9)
    
    score = _safe_divide(pi, cost_safe, eps=1e-9)
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
