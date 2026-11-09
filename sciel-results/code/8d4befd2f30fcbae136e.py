import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we've exceeded 320s, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: simple UCB-like score
        sigma_safe = np.maximum(sigma, 1e-9)
        score = -mu + 1.0 * sigma_safe
        score /= np.maximum(cost, 1e-9)
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Use a dynamic exploration weight that decreases as budget depletes
    # Estimate progress: cost is the cost to evaluate this point
    # Total budget is 30, initial 5 points cost ~5*1.5=7.5, so remaining ~22.5
    # Use cost as a proxy for how expensive this point is
    # Higher cost points should get more exploration bonus
    
    # Simple adaptive strategy: scale UCB weight by sqrt of sigma
    # This gives more weight to uncertain regions
    kappa = 1.5 + 0.5 * np.log1p(np.mean(sigma_safe))
    
    # UCB score (maximization): -mu + kappa * sigma
    ucb = -mu + kappa * sigma_safe
    
    # Blend EI and UCB
    # Weight EI more when it's large (exploitation)
    # Weight UCB more when sigma is large (exploration)
    ei_weight = 0.5
    ucb_weight = 0.5
    
    score = ei_weight * ei + ucb_weight * ucb
    
    # Cost-aware normalization
    score /= np.maximum(cost, 1e-9)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
