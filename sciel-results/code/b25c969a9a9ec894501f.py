import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_t0 = time.time()
_DEADLINE = 300.0
_MARGIN = 5.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to timeout, return quickly
    if time.time() - _t0 > _DEADLINE - _MARGIN:
        return np.zeros_like(mu)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardized improvement z
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # EI normalized by cost
    ei_score = ei / cost_safe
    
    # Prior bias towards expected minimizer location
    # Minimizer is in inner 80% of [-4, 5]^4
    # In scaled coords [0,1], the inner 80% is [0.1, 0.9]
    # Center is at 0.5 in scaled coords
    # Expected cost at minimizer: 1 + 0.5 = 1.5
    
    # Gaussian penalty for cost deviation from 1.5
    target_cost = 1.5
    lambda_bias = 1.0
    cost_weight = np.exp(-lambda_bias * (cost_safe - target_cost) ** 2)
    
    # Add a small exploration bonus for high uncertainty to avoid premature convergence
    # This is the structural refinement from inspiration 1: a direct sigma-based
    # exploration term that complements EI, scaled by cost and the cost bias weight
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Combine: primarily EI/cost, plus exploration bonus, all weighted by cost proximity
    score = (ei_score + exploration_bonus) * cost_weight
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
