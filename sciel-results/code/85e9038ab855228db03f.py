import numpy as np
from scipy.stats import norm
import time

# Global state to track evaluation count for dynamic weighting
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI)
    and a penalty on uncertainty (sigma) to balance exploration and exploitation.
    
    Args:
        mu: array of shape (N,) with posterior means.
        sigma: array of shape (N,) with posterior standard deviations.
        f_best: float, the best value observed so far.
        
    Returns:
        array of shape (N,) with scores. Higher scores indicate points 
        that are more likely to be the next best evaluation.
    """
    global _eval_count
    
    # Increment evaluation counter
    _eval_count += 1
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Use log-space or direct calculation? Direct is usually fine for double precision.
    # Phi(z) = norm.cdf(z)
    # phi(z) = norm.pdf(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Exploration Weight
    # We have 30 evaluations total.
    # Early on (low _eval_count), we want to explore more (high weight on sigma).
    # Later on (high _eval_count), we want to exploit more (low weight on sigma).
    
    # A simple decay: weight = 1 / (1 + _eval_count / 5)
    # Or exponential: weight = exp(-_eval_count / 10)
    # Let's use a linear decay from 1.0 to ~0.2 over the 30 steps.
    # weight = 1.0 - 0.8 * (_eval_count / 30.0)
    
    # Alternative: Use a fixed weight that has worked well in similar tasks.
    # Often, a weight around 0.1-0.5 on normalized sigma works.
    # Since sigma is in objective units, and f_best is in objective units,
    # we need to be careful with scaling.
    # Let's normalize sigma by a characteristic scale of the problem or f_best magnitude.
    # Goldstein-Price range is roughly 3 to ~100. f_best starts high, drops to 3.
    # Let's normalize sigma by max(f_best, 1.0) to make the penalty scale-invariant.
    
    scale = max(abs(f_best), 1.0)
    norm_sigma = sigma / scale
    
    # Exploration weight: Start high, decrease.
    # At step 1, weight=1.0. At step 30, weight=0.1.
    weight = 1.0 - 0.9 * min(_eval_count / 30.0, 1.0)
    
    # Score = EI - weight * norm_sigma
    # Higher EI is better. Higher sigma is worse (penalty).
    scores = ei - weight * norm_sigma
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
