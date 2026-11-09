"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto-frontier hypervolume contribution.
    
    This approach models the budget-constrained search as a multi-objective optimization
    problem where we maximize Expected Improvement (EI) while minimizing evaluation Cost.
    Instead of a fixed weighted sum or division, we identify the Pareto front of (cost, EI)
    pairs and assign scores based on the hypervolume contribution of each point on the front.
    This explicitly balances the trade-off: high-EI points are favored if they don't cost
    disproportionately more than lower-EI alternatives.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma has a small floor to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Numerical stability
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Check deadline early
    if time.time() - t_start > _DEADLINE:
        return np.zeros_like(ei)
    
    # Identify Pareto Front (Minimize Cost, Maximize EI)
    # Sort indices by cost
    sort_idx = np.argsort(cost)
    sorted_cost = cost[sort_idx]
    sorted_ei = ei[sort_idx]
    
    # Find indices on the Pareto front
    # A point is on the front if its EI is strictly greater than the max EI of all points with lower cost
    # We iterate through sorted costs. Since costs can be equal, we must be careful.
    # Standard definition: A point (c, e) is non-dominated if there is no other point (c', e') 
    # such that c' <= c and e' >= e, with at least one inequality strict.
    # In sorted order by cost, a point is on the front if its EI is greater than the 
    # maximum EI seen so far (from points with strictly lower cost). 
    # However, if multiple points have the same cost, only the one with the highest EI is on the front.
    
    front_indices = []
    max_ei_so_far = -np.inf
    current_cost = -np.inf
    
    for i in range(len(sort_idx)):
        c_i = sorted_cost[i]
        e_i = sorted_ei[i]
        
        # If cost is strictly greater than current tracked cost, update max_ei_so_far logic
        # Actually, simpler: iterate, keep track of max EI for costs <= current.
        # But since we are sorting, we can just check if e_i > max_ei_so_far.
        # If costs are equal, the first one encountered (lowest index in sort, arbitrary) might not be the max EI.
        # To be robust against equal costs, we should group by cost or handle ties.
        # Given N=2000, a linear pass with tie handling is fine.
        
        # A point is on the Pareto front if its EI is strictly greater than the maximum EI 
        # of all points with cost <= c_i, excluding itself if it's not the max for that cost.
        # Simpler robust approach:
        # 1. Sort by cost.
        # 2. For each unique cost, find the max EI.
        # 3. The Pareto front consists of (cost, max_ei_at_cost) where max_ei_at_cost > max(max_ei_at_lower_cost).
        
        pass # The loop above is a bit tricky with ties. Let's use the vectorized/robust method below.
    
    # Robust Pareto Front Extraction
    # 1. Get unique costs and their max EI
    unique_costs, inverse_indices = np.unique(sorted_cost, return_inverse=True)
    # Map back to original indices? No, we just need the front points to assign scores.
    # We need to assign scores to the original 2000 points.
    # Only points on the front get non-zero scores (or high scores).
    
    # Let's find the max EI for each unique cost
    max_ei_per_cost = np.full(len(unique_costs), -np.inf)
    # We need the index in the original array for the point that achieves this max EI at each cost.
    # Actually, we just need to identify which points are on the front.
    
    # Efficient way:
    # Sort by cost. 
    # Iterate through sorted array. Keep track of `global_max_ei`.
    # If current EI > global_max_ei, it's a new Pareto point. Update global_max_ei.
    # If current EI == global_max_ei and cost is same as previous, it's dominated by the previous one (or equivalent).
    # If current EI < global_max_ei, it's dominated.
    
    # Wait, if cost is the same, and EI is the same, they are equivalent. 
    # If cost is same, and EI is lower, dominated.
    # If cost is higher, and EI is lower, dominated.
    # If cost is higher, and EI is higher, new front point.
    
    # Let's do it cleanly:
    sort_idx = np.argsort(cost)
    sorted_cost = cost[sort_idx]
    sorted_ei = ei[sort_idx]
    
    is_on_front = np.zeros(len(sort_idx), dtype=bool)
    max_ei_so_far = -np.inf
    
    for i in range(len(sort_idx)):
        if sorted_ei[i] > max_ei_so_far:
            is_on_front[i] = True
            max_ei_so_far = sorted_ei[i]
        # If sorted_ei[i] == max_ei_so_far, it's not strictly better, so not on front (dominated by or equivalent to previous).
        # This ensures only the "best" point at a given cost level (if costs are sorted, the first one with that EI is kept) is selected.
        # However, if costs are identical, the first one in sort order is arbitrary. 
        # To be fair, we should probably give the same score to all points with the same cost and EI on the front?
        # Or just pick one. Given the noise, picking one is fine.
        
    # Map back to original indices
    front_orig_indices = sort_idx[is_on_front]
    
    if len(front_orig_indices) < 5 or np.max(ei) < 1e-6:
        # Fallback: Cost-normalized EI with exploration
        cost_penalized = np.maximum(cost, 1e-6)
        sigma_max = np.max(sigma_safe)
        if sigma_max > 1e-9:
            exploration_bonus = 0.1 * sigma_safe / sigma_max * np.max(ei)
        else:
            exploration_bonus = 0.0
        scores = (ei + exploration_bonus) / cost_penalized
    else:
        scores = np.zeros_like(ei)
        
        # Calculate Hypervolume contribution for points on the Pareto front
        # The front points are sorted by cost in `front_orig_indices`? 
        # `sort_idx` was sorted by cost. `is_on_front` was applied in that order.
        # So `front_orig_indices` is already in increasing cost order.
        
        f_costs = cost[front_orig_indices]
        f_eis = ei[front_orig_indices]
        
        # c_next is the cost of the next point on the front, or max(cost) for the last one
        c_nexts = np.empty_like(f_costs)
        if len(f_costs) > 1:
            c_nexts[:-1] = f_costs[1:]
        c_nexts[-1] = np.max(cost)
        
        # HV contribution = width * height
        # Width = c_next - c_current
        # Height = EI_current (reference EI is 0)
        hv_contribs = (c_nexts - f_costs) * f_eis
        
        # Assign scores
        scores[front_orig_indices] = hv_contribs
        
        # Add a tiny exploration bonus to break ties and encourage diversity on the front
        # Scale by max EI to keep it relative
        if np.max(f_eis) > 0:
            scores[front_orig_indices] += 1e-6 * (sigma_safe[front_orig_indices] / np.max(sigma_safe)) * np.max(f_eis)
        
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
