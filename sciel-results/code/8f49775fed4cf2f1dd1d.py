"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-frontier-aware acquisition: computes EI and cost, then scores each
    candidate by its hypervolume contribution to the non-dominated (cost, EI)
    front. This explicitly models the multi-objective trade-off rather than
    using a fixed weighted sum.
    """
    n = len(mu)
    eps = 1e-9
    
    # Compute expected improvement
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, eps)
    
    # Build (cost, ei) pairs - we want to maximize EI and minimize cost
    # Non-dominated: no other point has both >= EI and <= cost
    # For hypervolume contribution, we work in the "maximize EI, minimize cost" space
    # Transform: x = cost (minimize), y = ei (maximize)
    
    # Find non-dominated points
    # A point i is dominated if there exists j with cost[j] <= cost[i] AND ei[j] >= ei[i]
    # (with at least one strict)
    
    # For efficiency with 2000 points, use a sorted approach
    # Sort by cost ascending, then track max EI seen so far
    order = np.argsort(cost)
    sorted_cost = cost[order]
    sorted_ei = ei[order]
    
    # For each point, determine if it's on the Pareto front
    # and compute its hypervolume contribution
    # 
    # Hypervolume contribution of point i on the front:
    # It dominates the rectangle from (cost[i], 0) to (next_cost_on_front, ei[i])
    # where next_cost_on_front is the cost of the next point on the front with higher cost
    # and lower EI (or infinity for the last point)
    
    # Build the Pareto front: sort by cost asc, keep points where EI is strictly increasing
    # (since we want max EI, min cost)
    front_idx = []
    max_ei_so_far = -np.inf
    for i in range(n):
        if sorted_ei[i] > max_ei_so_far + eps:
            front_idx.append(i)
            max_ei_so_far = sorted_ei[i]
    
    if len(front_idx) == 0:
        # Fallback: return normalized EI
        return ei / cost
    
    # Compute hypervolume contribution for each front point
    # The reference point is (max_cost, 0) for the dominated region
    # Point i contributes: (cost[i+1] - cost[i]) * ei[i] where cost[i+1] is next front point's cost
    # Last front point contributes: (max_cost - cost[i]) * ei[i]
    
    hv_contrib = np.zeros(n)
    max_cost_val = np.max(sorted_cost)
    
    for k, idx in enumerate(front_idx):
        c = sorted_cost[idx]
        e = sorted_ei[idx]
        if k < len(front_idx) - 1:
            next_c = sorted_cost[front_idx[k + 1]]
        else:
            next_c = max_cost_val + 1.0  # extend beyond max for the last point
        
        width = next_c - c
        if width < 0:
            width = 0
        hv_contrib[idx] = width * e
    
    # For non-front points, assign 0 contribution (they're dominated)
    # But we need to return scores for ALL 2000 candidates
    # Non-front points get a score proportional to how close they are to the front
    
    # Map back to original indices
    hv_orig = np.zeros(n)
    hv_orig[order] = hv_contrib
    
    # For points not on the front, assign a small score based on their EI/cost ratio
    # scaled relative to the minimum front hypervolume
    front_mask = hv_orig > 0
    if np.any(front_mask):
        min_hv = np.min(hv_orig[front_mask])
        # Non-front points get a fraction of their EI/cost, scaled down
        non_front_score = ei / cost
        # Scale non-front scores to be below the minimum front score
        non_front_score = non_front_score * min_hv * 0.5 / (np.max(non_front_score) + eps)
    else:
        non_front_score = ei / cost
    
    result = np.where(front_mask, hv_orig, non_front_score)
    
    # Ensure all finite and positive
    result = np.where(np.isfinite(result), result, 0.0)
    result = np.maximum(result, 0.0)
    
    # Normalize to [0, 1] range for stability
    max_val = np.max(result)
    if max_val > 0:
        result = result / max_val
    
    return result
# EVOLVE-BLOCK-END
