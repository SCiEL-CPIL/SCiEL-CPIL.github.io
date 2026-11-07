"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement (EI) for minimization
    # Improvement is defined as f_best - f(x)
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # UCB = mu - kappa * sigma
    # Lower UCB means more promising
    kappa = 2.0
    ucb = mu - kappa * sigma
    
    # Normalize EI and UCB to [0, 1] for fair comparison
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        # Invert: lower UCB = higher norm
        ucb_norm = 1.0 - (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
        
    # Combine EI and UCB into a single "Quality" metric
    # We use the weights from the current program which provided a stable baseline
    weight_ei = 0.6
    weight_ucb = 0.4
    
    quality = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Rank-based Pareto front selection (Strategy from Inspiration 1)
    # A candidate is non-dominated if no other candidate has both
    # higher quality AND lower cost
    
    n = len(quality)
    if n == 0:
        return np.array([])
        
    # Efficient O(n log n) Pareto front detection
    # Sort by cost ascending. For each position, check if any earlier 
    # (lower cost) candidate has higher quality.
    
    cost_order = np.argsort(cost)
    sorted_cost = cost[cost_order]
    sorted_quality = quality[cost_order]
    
    # Running max of quality from left (lower cost side)
    # A point at position i is dominated if there exists a point j < i 
    # such that cost[j] < cost[i] (guaranteed by sort) AND quality[j] > quality[i]
    # So we check if the max quality seen so far (excluding current) is > current quality.
    
    is_dominated = np.zeros(n, dtype=bool)
    
    if n > 1:
        # running_max[i] is the max quality among indices 0..i
        running_max = np.zeros(n)
        running_max[0] = sorted_quality[0]
        
        for i in range(1, n):
            # If the max quality seen in all previous (lower cost) points is strictly greater 
            # than the current point's quality, it is dominated.
            # Note: We use the running_max from the previous step (i-1) to check against current (i)
            # However, a simpler vectorized approach or loop is fine for n=2000.
            # Let's do it carefully:
            # is_dominated[i] = (max(sorted_quality[0:i]) > sorted_quality[i])
            
            # We can compute this iteratively.
            running_max[i] = max(running_max[i-1], sorted_quality[i])
            if running_max[i-1] > sorted_quality[i]:
                is_dominated[i] = True
                
        # Map back to original indices
        # is_dominated is in sorted order, so we need to unsort it
        dominated_mask_original = np.zeros(n, dtype=bool)
        dominated_mask_original[cost_order] = is_dominated
        non_dominated_mask = ~dominated_mask_original
    else:
        non_dominated_mask = np.ones(n, dtype=bool)
        
    # Assign scores
    final_score = np.zeros(n)
    
    if np.any(non_dominated_mask):
        nd_indices = np.where(non_dominated_mask)[0]
        # Rank non-dominated by their quality score
        nd_quality = quality[nd_indices]
        if len(nd_quality) > 1:
            # Higher quality gets higher rank (0 to 1)
            nd_rank = np.argsort(np.argsort(nd_quality)) / (len(nd_quality) - 1)
        else:
            nd_rank = np.ones_like(nd_quality)
        
        # Non-dominated points get scores in [0.6, 1.0]
        # This ensures they are always preferred over dominated points
        final_score[nd_indices] = 0.6 + 0.4 * nd_rank
    
    if np.any(~non_dominated_mask):
        dom_indices = np.where(~non_dominated_mask)[0]
        # Rank dominated by their quality score, but give lower scores
        dom_quality = quality[dom_indices]
        if len(dom_quality) > 1:
            dom_rank = np.argsort(np.argsort(dom_quality)) / (len(dom_quality) - 1)
        else:
            dom_rank = np.ones_like(dom_quality)
            
        # Dominated points get scores in [0.0, 0.3]
        final_score[dom_indices] = 0.3 * dom_rank
        
    # Ensure finite output
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_score
# EVOLVE-BLOCK-END
