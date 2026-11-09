# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
DEADLINE_MARGIN = 30.0

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Cost-Adjusted LCB.
    """
    if _elapsed() > DEADLINE_MARGIN:
        # Fallback: Simple EI/Cost
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / (cost + 1e-12)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive Beta: High exploration when f_best is large, low when small
    # Using a smooth saturation function
    f_best_val = float(f_best)
    beta = 1.0 + 2.0 * (f_best_val / (f_best_val + 10.0))
    
    # Lower Confidence Bound for minimization: mu - beta * sigma
    # We want to maximize the potential improvement: f_best - (mu - beta * sigma)
    lcb = mu - beta * sigma_safe
    improvement_potential = f_best - lcb
    
    # Cost-aware score: Improvement per unit cost
    score = improvement_potential / (cost + 1e-12)
    
    # Ensure non-negative and finite
    score = np.maximum(score, 0.0)
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
