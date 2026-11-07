import numpy as np
from scipy.stats import norm
import time

_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a cost-aware scaling that
    rewards points with high expected improvement relative to their cost.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 300s limit, return a fast fallback
    elapsed = _get_elapsed()
    if elapsed > 280:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / (cost + 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-scores for Expected Improvement (Minimization)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scaling using linear inverse-cost weighting (EI / cost)
    # This maximizes improvement efficiency per unit cost.
    # Cost ranges from 1 to 2, so 1/cost ranges from 0.5 to 1.0.
    # This is a structural change from multiplicative Gaussian penalty to
    # maximizing improvement per unit cost, which better balances the
    # exploration-exploitation tradeoff under the cost constraint.
    
    cost_safe = np.maximum(cost, 1e-6)
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
