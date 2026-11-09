import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Check deadline
    if _check_deadline():
        # Fallback: Simple EI / Cost
        sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = np.maximum((f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z), 0.0)
        return ei / (cost + 1e-9)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Normalize EI and Cost to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    
    ei_range = ei_max - ei_min
    cost_range = cost_max - cost_min
    
    # Avoid division by zero
    if ei_range < 1e-12:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
        
    # 3. Identify Non-Dominated Points (Pareto Front)
    # Objective: Maximize EI, Minimize Cost
    # Sort by EI descending
    sorted_indices = np.argsort(-ei)
    
    is_non_dominated = np.zeros(len(ei), dtype=bool)
    min_cost_so_far = np.inf
    
    for idx in sorted_indices:
        if cost[idx] < min_cost_so_far:
            is_non_dominated[idx] = True
            min_cost_so_far = cost[idx]
        # If cost[idx] == min_cost_so_far, it is dominated by the previous one with higher/equal EI
        # (Since we sort by EI descending, previous ones have >= EI. If cost is same, previous dominates current)
        
    # 4. Calculate Scores based on Hypervolume Contribution Proxy
    # Score = EI_norm * (1 - Cost_norm) for non-dominated points
    # This approximates the area contribution in the normalized 2D objective space
    scores = np.zeros(len(ei))
    
    # Only assign scores to non-dominated points
    scores[is_non_dominated] = ei_norm[is_non_dominated] * (1.0 - cost_norm[is_non_dominated])
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
