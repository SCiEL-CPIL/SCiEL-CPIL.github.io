"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using hypervolume-contribution-based
    multi-objective scoring over (cost, expected improvement) pairs.
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Build the objective vector: minimize cost, maximize EI
    # Represent as (cost, -EI) for minimization
    obj = np.column_stack([cost, -ei])

    # Find non-dominated points in (cost, -ei) space
    n = len(obj)
    nondominated = _find_nondominated(obj)

    # Hypervolume contribution of each point relative to the reference point
    # Reference point: worst cost and worst -EI (i.e., max cost, min EI)
    ref_cost = np.max(obj[:, 0]) + 1e-6
    ref_neg_ei = np.min(obj[:, 1]) - 1e-6  # i.e., ref in -EI space
    # In minimization of (cost, -ei), ref is (ref_cost, ref_neg_ei)
    # Hypervolume contribution for a point p_i = (c_i, ne_i):
    # For 2D, the contribution is the area of the rectangle [c_i, ref_cost] x [ne_i, ref_neg_ei]
    # but only counting the part not already covered by other nondominated points.
    # A simpler approximation: for each point, its "marginal" hypervolume is
    # the area of the box it uniquely dominates. For 2D with sorted nondominated points,
    # we can compute this exactly.

    # Compute hypervolume contribution for each candidate
    hv_scores = np.zeros(n, dtype=np.float64)

    # Get the sorted nondominated points by cost (ascending)
    if len(nondominated) > 0:
        nd_idx = np.where(nondominated)[0]
        # Sort by cost ascending
        sort_order = np.argsort(obj[nd_idx, 0])
        nd_sorted = nd_idx[sort_order]

        # For 2D, hypervolume contribution of point i in the sorted nondominated set:
        # width = cost[i+1] - cost[i] (or ref_cost - cost[i] for the last)
        # height = ref_neg_ei - (-EI[i])  ... wait, in minimization:
        # The dominated region of point i is [c_i, c_{i+1}] x [ne_i, ne_{i+1}] ... 
        # Actually for 2D HV, the contribution of point i (sorted by c ascending) is:
        # (c_{i+1} - c_i) * (ref_ne - ne_i)  where c_{i+1} is the next point's cost or ref
        # and ne_i is this point's -EI value.
        
        # But we want to score ALL points, not just nondominated ones.
        # For dominated points, their contribution is 0.
        # For nondominated points, compute their marginal HV.
        
        for pos in range(len(nd_sorted)):
            idx = nd_sorted[pos]
            c_i = obj[idx, 0]
            ne_i = obj[idx, 1]
            
            if pos < len(nd_sorted) - 1:
                c_next = obj[nd_sorted[pos + 1], 0]
            else:
                c_next = ref_cost
            
            c_next = max(c_next, c_i + 1e-12)
            width = c_next - c_i
            height = ref_neg_ei - ne_i
            height = max(height, 0.0)
            
            hv_scores[idx] = width * height
    
    # Scale HV scores to be comparable and positive
    # Add a small EI component to break ties and guide exploration
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    # Combine: hypervolume contribution + small EI term
    # The HV score already captures both objectives, but we add EI to ensure
    # that among points with similar HV, higher EI is preferred
    scores = hv_scores + 0.05 * ei_norm * ei_max

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores

    return scores


def _find_nondominated(obj):
    """
    Find indices of non-dominated points in a 2D minimization problem.
    A point is non-dominated if no other point has both coordinates <= it
    with at least one strict inequality.
    
    Parameters:
        obj: array of shape (n, 2)
    
    Returns:
        boolean mask of shape (n,)
    """
    n = len(obj)
    nondom = np.ones(n, dtype=bool)
    
    # For each point, check if it is dominated by any other
    # Point i is dominated by point j if obj[j,0] <= obj[i,0] and obj[j,1] <= obj[i,1]
    # and (obj[j,0] < obj[i,0] or obj[j,1] < obj[i,1])
    
    # Efficient 2D approach: sort by first coordinate, then track minimum of second
    order = np.argsort(obj[:, 0])
    sorted_obj = obj[order]
    
    # A point is non-dominated if its second coordinate is strictly less than
    # the minimum second coordinate seen so far (from points with smaller or equal first coord)
    min_second = np.inf
    for i in range(n):
        c = sorted_obj[i, 0]
        s = sorted_obj[i, 1]
        if s < min_second:
            min_second = s
            # Mark this point as potentially non-dominated
            # But we need to handle ties in first coordinate
        else:
            # This point is dominated by a previous point with <= first coord and < second coord
            # Actually we need to be more careful with ties
            pass
    
    # Better approach for 2D: 
    # Sort by first coordinate ascending, then for ties, sort by second coordinate ascending.
    # A point is non-dominated if no point before it has second coordinate <= its second.
    
    order = np.lexsort((obj[:, 1], obj[:, 0]))
    sorted_obj = obj[order]
    
    # Track running minimum of second coordinate
    running_min = np.empty(n, dtype=np.float64)
    running_min[0] = sorted_obj[0, 1]
    for i in range(1, n):
        running_min[i] = min(running_min[i - 1], sorted_obj[i, 1])
    
    # A point is non-dominated if its second coordinate equals the running min up to that point
    # AND it is not strictly dominated by a point with the same first coordinate and lower second
    nondom_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if sorted_obj[i, 1] == running_min[i]:
            nondom_sorted[i] = True
    
    # Map back to original indices
    nondom = np.zeros(n, dtype=bool)
    for i in range(n):
        if nondom_sorted[i]:
            nondom[order[i]] = True
    
    return nondom


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
