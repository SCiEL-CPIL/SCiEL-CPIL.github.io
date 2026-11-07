"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _non_dominated_front_2d(a, b):
    """
    Find indices of non-dominated points in 2D where both a and b are to be maximized.
    A point i dominates j if a[i] >= a[j] and b[i] >= b[j] and (a[i] > a[j] or b[i] > b[j]).
    Returns boolean mask of non-dominated points.
    """
    n = len(a)
    if n == 0:
        return np.array([], dtype=bool)
    
    # Sort by a descending, then b descending
    order = np.lexsort((-b, -a))
    sorted_a = a[order]
    sorted_b = b[order]
    
    # A point is dominated if there exists a previous point with both coordinates >=
    # Since sorted by a desc, for each point, we need to check if max b so far >= current b
    # But we need strict dominance in at least one coordinate.
    # Actually, non-dominated: no other point has a >= current_a and b >= current_b with at least one strict.
    # In sorted order (a desc), for point i, all previous points have a >= a[i].
    # If max(b[0..i-1]) >= b[i], then point i is dominated (by the one with max b, since a is >=).
    # Wait, if a[j] == a[i] and b[j] == b[i], they dominate each other? 
    # Usually in Pareto front, we keep one or all. Let's keep all that are not strictly dominated.
    # Strictly dominated: exists j such that a[j] >= a[i], b[j] >= b[i], and (a[j] > a[i] or b[j] > b[i]).
    
    # Let's use a simpler O(n^2) approach for n=2000? 2000^2 = 4 million. That's fast in NumPy if vectorized.
    # But we can do better.
    
    # Let's try the sorting approach carefully.
    # Sort by a descending.
    # For each point in sorted order, maintain max_b_seen_so_far.
    # If current b > max_b_seen_so_far, it's non-dominated (no previous point has b >= current b, since max was lower).
    # If current b <= max_b_seen_so_far, it IS dominated?
    #   Previous point j has a[j] >= a[i] (since sorted).
    #   And b[j] >= max_b_seen_so_far? No, max_b_seen_so_far is the max of b[0..i-1].
    #   So there exists j < i with b[j] = max_b_seen_so_far >= current b.
    #   And a[j] >= a[i].
    #   If a[j] > a[i] or b[j] > current b, then j strictly dominates i.
    #   If a[j] == a[i] and b[j] == current b, they are equivalent.
    
    # Let's just compute the non-dominated front using vectorized operations for 2000 points.
    # 2000x2000 matrix is 4M entries. That's fine.
    
    # a_i >= a_j and b_i >= b_j
    a_i = a[:, None]
    a_j = a[None, :]
    b_i = b[:, None]
    b_j = b[None, :]
    
    # i dominates j if a_i >= a_j and b_i >= b_j and (a_i > a_j or b_i > b_j)
    dominates = (a_i >= a_j) & (b_i >= b_j) & ((a_i > a_j) | (b_i > b_j))
    
    # A point is non-dominated if no other point dominates it.
    # dominated[i] = any(dominates[:, i])
    dominated = dominates.any(axis=0)
    
    return ~dominated


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Compute Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Uncertainty (UCB-like term)
    # Use a fixed beta or adaptive? Let's use a fixed beta for now, or maybe 1.0.
    # Actually, let's use the standard deviation itself as the uncertainty measure.
    uncertainty = sigma
    
    # Cost is given. We want to minimize cost, so higher cost is worse.
    # For Pareto front, we want to maximize (EI, Uncertainty) and minimize Cost.
    # So let's define the third dimension as -cost or 1/cost.
    # Let's use 1/cost to make it a maximization problem.
    cost_inv = 1.0 / (cost + 1e-9)
    
    # We have 3 dimensions: EI, Uncertainty, Cost_inv
    # We want to find the Pareto front in 3D.
    # But the directive says: "filters candidates based on a non-dominated sort of (predicted improvement, uncertainty, cost)"
    # So we have 3 objectives.
    # Let's implement a simple 3D non-dominated front finder.
    # For 2000 points, O(n^2) is 4M operations per dimension pair? No, it's O(n^2) total if we check all pairs.
    # 2000^2 = 4M. That's fast.
    
    # Let's check time.
    if time.time() > deadline:
        # Fallback to EI/cost
        return ei / cost
    
    # 3D Non-dominated front
    # i dominates j if:
    # ei[i] >= ei[j] and unc[i] >= unc[j] and cost_inv[i] >= cost_inv[j]
    # and at least one is strict.
    
    ei_i = ei[:, None]
    ei_j = ei[None, :]
    unc_i = uncertainty[:, None]
    unc_j = uncertainty[None, :]
    cost_i = cost_inv[:, None]
    cost_j = cost_inv[None, :]
    
    # This will create 3 matrices of size 2000x2000 = 4M each.
    # Total memory: 3 * 4M * 8 bytes = 96 MB. That's acceptable.
    
    dominates = ((ei_i >= ei_j) & (unc_i >= unc_j) & (cost_i >= cost_j) & 
                 ((ei_i > ei_j) | (unc_i > unc_j) | (cost_i > cost_j)))
    
    if time.time() > deadline:
        return ei / cost
    
    dominated = dominates.any(axis=0)
    on_front = ~dominated
    
    # For points on the front, we need a score.
    # The directive says: "selecting only those on the efficient frontier to balance exploration and cost without arbitrary weight tuning."
    # So we return a high score for points on the front, and 0 (or very low) for others.
    # But we need to rank them. Let's use a normalized score based on their position on the front.
    # Or simply return the EI value for points on the front, and 0 for others.
    # Or return a combined score like (ei + uncertainty + cost_inv) for points on the front.
    
    # Let's return the EI value for points on the front, scaled by a large factor to ensure they are selected.
    # But if multiple points are on the front, we need to pick the best one.
    # The evaluator picks the argmax. So if we return EI for front points, it will pick the one with highest EI.
    # That might be too exploitative.
    
    # Let's return a score that balances the three objectives.
    # Since they are non-dominated, we can use a simple sum or product.
    # Let's use the geometric mean or product.
    # Or just return the maximum of the three normalized objectives.
    
    # Let's normalize each objective to [0, 1] among the front points.
    front_indices = np.where(on_front)[0]
    if len(front_indices) == 0:
        # Fallback
        return ei / cost
    
    front_ei = ei[front_indices]
    front_unc = uncertainty[front_indices]
    front_cost = cost_inv[front_indices]
    
    # Normalize
    ei_min, ei_max = front_ei.min(), front_ei.max()
    unc_min, unc_max = front_unc.min(), front_unc.max()
    cost_min, cost_max = front_cost.min(), front_cost.max()
    
    ei_norm = (front_ei - ei_min) / (ei_max - ei_min + 1e-9)
    unc_norm = (front_unc - unc_min) / (unc_max - unc_min + 1e-9)
    cost_norm = (front_cost - cost_min) / (cost_max - cost_min + 1e-9)
    
    # Score is the sum of normalized objectives
    front_score = ei_norm + unc_norm + cost_norm
    
    # Create the final score array
    scores = np.zeros_like(mu)
    scores[front_indices] = front_score
    
    # Scale to ensure these are the top candidates
    # Add a large offset to ensure they are above all non-front points
    scores[front_indices] += 1000.0
    
    return scores
# EVOLVE-BLOCK-END
