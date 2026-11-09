import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Shekel 4D.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed function value so far
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    global _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    elapsed = time.time() - _start_time
    if elapsed > 325:
        # Safety: return exploit-heavy fallback
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / (cost_safe + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-9, 1e6)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI (maximize): use log-sum-exp style smooth approach
    # The idea: we want to maximize EI while minimizing cost
    # Use a weighted combination with adaptive scaling
    
    n = len(ei)
    
    # Normalize EI to [0, 1]
    ei_min, ei_max = ei.min(), ei.max()
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.ones(n)
    
    # Normalize cost to [0, 1] (inverted: lower cost = higher score)
    cost_min, cost_max = cost_safe.min(), cost_safe.max()
    if cost_max > cost_min:
        cost_norm = 1.0 - (cost_safe - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones(n)
    
    # Also consider the expected improvement per unit cost
    ei_per_cost = ei / cost_safe
    epc_min, epc_max = ei_per_cost.min(), ei_per_cost.max()
    if epc_max > epc_min:
        epc_norm = (ei_per_cost - epc_min) / (epc_max - epc_min)
    else:
        epc_norm = np.ones(n)
    
    # Combine: weight EI, cost efficiency, and EI/cost
    # Use a geometric mean approach for smoothness (avoids hard Pareto selection)
    # Geometric mean of normalized components
    eps = 1e-10
    scores = np.power(
        (ei_norm + eps) * (cost_norm + eps) * (epc_norm + eps),
        1.0 / 3.0
    )
    
    # Add a small exploitation bonus for candidates with very low mu
    # (these are likely near the optimum)
    mu_min, mu_max = mu.min(), mu.max()
    if mu_max > mu_min:
        mu_norm = 1.0 - (mu - mu_min) / (mu_max - mu_min)  # lower mu = higher
    else:
        mu_norm = np.ones(n)
    
    # Blend: 80% geometric mean, 20% exploitation
    scores = 0.8 * scores + 0.2 * mu_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
