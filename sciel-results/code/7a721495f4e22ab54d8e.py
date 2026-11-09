"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def _pareto_front_indices(ei, cost):
    """
    Identify the indices of the non-dominated set (Pareto front) for 
    maximizing ei and minimizing cost.
    
    A point i is dominated by point j if:
    ei[j] >= ei[i] AND cost[j] <= cost[i] AND (ei[j] > ei[i] OR cost[j] < cost[i])
    
    We return indices of points NOT dominated by any other point.
    For performance with N=2000, we use a vectorized check.
    """
    n = len(ei)
    if n == 0:
        return np.array([], dtype=int)
    
    # Sort by ei descending to speed up checks (optional optimization, but vectorized is fine for 2000)
    # Vectorized dominance check:
    # Dominated if exists j != i such that ei[j] >= ei[i] and cost[j] <= cost[i] and (ei[j] > ei[i] or cost[j] < cost[i])
    
    # To handle strict inequalities correctly in vectorized ops:
    # Let's create masks.
    # For each i, check if there is a j that dominates it.
    
    # Efficient approach:
    # 1. Sort by EI descending.
    # 2. Keep track of minimum cost seen so far.
    # 3. A point is on the front if its cost is lower than the minimum cost seen so far (from higher EI points).
    
    order = np.argsort(ei)[::-1]  # Descending EI
    sorted_ei = ei[order]
    sorted_cost = cost[order]
    
    # Initialize min_cost array
    # min_cost[i] is the minimum cost among points with EI >= sorted_ei[i]
    # We can compute this by iterating, but let's try vectorized cummin from the right?
    # No, cummin from left on reversed array works.
    
    # Let's just do a simple loop for clarity and safety, N=2000 is small enough for a Python loop?
    # 2000 iterations is ~0.1ms. Very fast.
    
    is_front = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    
    # We need to map back to original indices.
    # Let's create an array of indices sorted by EI desc
    # And iterate through them.
    
    # However, ties in EI need care. If EI is same, lower cost is better.
    # The standard "skyline" algorithm:
    # Sort by EI desc. If tie, sort by cost asc.
    # Then, a point is on the front if its cost is strictly less than the min cost of all preceding points (which have higher or equal EI).
    
    # Let's sort properly
    # Sort by EI descending, then cost ascending
    # np.lexsort uses the last key as primary. 
    # We want primary: EI desc, secondary: cost asc.
    # So sort by (-EI, cost).
    
    keys = (-ei, cost)
    sorted_idx = np.lexsort(keys)
    
    sorted_ei = ei[sorted_idx]
    sorted_cost = cost[sorted_idx]
    
    # Now iterate
    min_cost = np.inf
    front_indices_in_sorted = []
    
    # We can vectorize the "is new minimum" part?
    # A point is on the front if cost[i] < min(cost[0..i-1])
    # Let's compute the cumulative minimum of cost from the start (which corresponds to highest EI).
    # But we need strict inequality for "dominated"?
    # If cost[i] == min_cost_so_far, it is dominated by the previous one (same EI, lower/equal cost).
    # So we want cost[i] < min_cost_so_far.
    
    # Cumulative minimum of sorted_cost:
    # cum_min[i] = min(sorted_cost[0], ..., sorted_cost[i])
    # If sorted_cost[i] < cum_min[i-1], then it's a new minimum.
    
    # Let's compute cum_min
    # np.minimum.accumulate(sorted_cost)
    cum_min = np.minimum.accumulate(sorted_cost)
    
    # A point i is on the front if sorted_cost[i] < min(sorted_cost[0:i])
    # Note: for i=0, min of empty set is inf, so cost[0] < inf is True.
    
    # Create a shifted cum_min to represent "min of previous elements"
    # prev_min = np.roll(cum_min, 1)
    # prev_min[0] = np.inf
    
    prev_min = np.empty_like(cum_min)
    prev_min[0] = np.inf
    prev_min[1:] = cum_min[:-1]
    
    # Condition: sorted_cost < prev_min
    is_front_sorted = sorted_cost < prev_min
    
    # Get original indices
    front_indices = sorted_idx[is_front_sorted]
    
    return front_indices

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front selection on (EI, Cost).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores. Higher scores indicate better candidates.
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI)
    # EI = E[max(0, f_best - f(x))]
    # f(x) ~ N(mu, sigma^2)
    # delta = (f_best - mu) / sigma
    # EI = sigma * (delta * Phi(delta) + phi(delta))
    
    delta = (f_best - mu) / sigma
    
    # Clip delta to avoid extreme values in cdf/pdf if necessary, though scipy handles it well
    # For very large positive delta, EI ~ f_best - mu
    # For very large negative delta, EI ~ 0
    
    cdf_delta = norm.cdf(delta)
    pdf_delta = norm.pdf(delta)
    
    ei = sigma * (delta * cdf_delta + pdf_delta)
    ei = np.maximum(ei, 0.0)
    
    # Identify Pareto front: Maximize EI, Minimize Cost
    # We want the set of points that are not dominated by any other point.
    # Dominance: j dominates i if EI[j] >= EI[i] and Cost[j] <= Cost[i] and (EI[j] > EI[i] or Cost[j] < Cost[i])
    
    front_indices = _pareto_front_indices(ei, cost)
    
    # Create scores
    # Points on the front get a high score.
    # Points not on the front get a low score (e.g., 0 or negative).
    # To differentiate among front points, we can score them by their EI/cost ratio or just give them all the same high score?
    # If we give them all the same score, the evaluator picks one randomly? No, it picks the max. 
    # If there are ties, it might pick the first one. 
    # It's better to rank them. 
    # Let's score front points by their "efficiency" or just EI.
    # Actually, the task says "score candidates based on their rank within this front".
    # Let's assign scores proportional to EI for front points, and 0 for others.
    # Or better, assign a high base score + EI to front points, and 0 to others.
    
    scores = np.zeros(len(ei), dtype=np.float64)
    
    if len(front_indices) > 0:
        # Assign high scores to front points
        # We want to prioritize points on the front.
        # Let's use EI as the score for front points, but ensure they are higher than any non-front point.
        # Since EI >= 0, and non-front are 0, this works if we add a small epsilon or just use EI.
        # However, if a non-front point has high EI but high cost, it's dominated.
        # A front point might have lower EI but much lower cost.
        # The "rank within front" instruction suggests we should differentiate.
        # Let's sort front points by EI/Cost ratio? Or just EI?
        # Let's stick to EI as the primary driver of value, but restrict to front.
        
        scores[front_indices] = ei[front_indices]
        
        # To ensure front points are strictly preferred over non-front points (which are 0),
        # and since EI can be 0 for some front points (if f_best is very low),
        # we might want to add a small bias to front points.
        # But if EI is 0, it's not very useful anyway.
        
        # Let's add a small constant to all front points to ensure they beat non-front points with EI=0?
        # Actually, if EI=0, the point is not expected to improve. 
        # Let's just use EI. If multiple front points have EI=0, it doesn't matter much.
        
        # To break ties and provide a "rank", we can subtract a small penalty based on cost?
        # The directive says "score candidates based on their rank within this front".
        # Let's sort the front points by their cost-normalized EI? 
        # Or simply by EI?
        # Let's use EI as the score.
        
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
