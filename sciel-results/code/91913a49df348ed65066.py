# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin before 150s hard limit

def _find_lambda(weights, values, capacity):
    """
    Find the Lagrangian multiplier lambda such that the sum of weights
    of items with positive reduced cost (v - lambda * w) is close to capacity.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
    
    # If total weight of all items is less than capacity, lambda is 0
    total_w = np.sum(weights)
    if total_w <= capacity:
        return 0.0
    
    # Binary search for lambda
    # Lower bound: 0
    # Upper bound: max density + epsilon
    densities = values / weights
    low = 0.0
    high = np.max(densities) + 1.0
    
    # We want to find lambda such that sum(w_i for v_i > lambda * w_i) ~= capacity
    # This is equivalent to sorting by density and finding the cutoff.
    # Actually, we can just sort by density and find the item that crosses the capacity threshold.
    # Let's do that, it's O(N log N) but simpler and exact for the relaxed problem.
    
    order = np.argsort(-densities)
    sorted_w = weights[order]
    sorted_v = values[order]
    
    cum_w = np.cumsum(sorted_w)
    
    # Find the index where cumulative weight exceeds capacity
    # np.searchsorted returns the index where the value would be inserted to maintain order
    # We want the first index where cum_w > capacity
    idx = np.searchsorted(cum_w, capacity, side='right')
    
    if idx == 0:
        # Even the first item is too heavy? Or capacity is 0?
        # If idx is 0, it means capacity < first weight? No, searchsorted returns 0 if capacity < cum_w[0]
        # In that case, lambda should be high.
        return high
        
    if idx == n:
        # All items fit
        return 0.0
        
    # The item at index idx-1 is the last one that fits fully.
    # The item at index idx is the one that would be fractional in the LP relaxation.
    # The lambda is the density of the item that is partially taken.
    # Specifically, if we take items 0..idx-1 fully, and item idx partially,
    # the lambda is the density of item idx.
    return densities[order[idx]]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Heuristic for 0/1 knapsack: pick the next item to add.
    Returns an index into the remaining (feasible) arrays, or -1 to stop.
    
    Strategy:
      1. Quick feasibility check.
      2. Compute Lagrangian multiplier lambda based on remaining capacity.
      3. Score items by reduced cost (v - lambda * w) or density.
      4. Select top candidates by a combination of Lagrangian score and density.
      5. 1-step look-ahead: simulate picking each candidate, then greedy fill by density.
      6. Return the candidate with the highest total value.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    w_arr = np.asarray(remaining_weights, dtype=np.float64)
    v_arr = np.asarray(remaining_values, dtype=np.float64)
    
    # Feasibility mask
    feasible = w_arr <= remaining_capacity
    if not np.any(feasible):
        return -1
    
    feasible_idxs = np.where(feasible)[0]
    if len(feasible_idxs) == 0:
        return -1
    
    # If only one feasible item, pick it
    if len(feasible_idxs) == 1:
        return int(feasible_idxs[0])
    
    # Check time budget
    if time.time() > _DEADLINE - 5.0:
        # Not enough time, use pure density
        densities = v_arr[feasible_idxs] / w_arr[feasible_idxs]
        best_local = int(np.argmax(densities))
        return int(feasible_idxs[best_local])
    
    # Compute densities for all items (for sorting in greedy fill)
    # Handle zero weights just in case, though usually weights > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        all_densities = np.divide(v_arr, w_arr, out=np.zeros_like(v_arr), where=w_arr>0)
        
    # Pre-sort all items by density for the greedy fill simulation
    # We only care about feasible items in the fill, but sorting all is easier
    density_order_all = np.argsort(-all_densities)
    
    # Find Lagrangian multiplier
    # We use the feasible items to estimate lambda for the remaining problem
    w_feas = w_arr[feasible_idxs]
    v_feas = v_arr[feasible_idxs]
    
    # If total weight of feasible items <= capacity, we can take all.
    # The "best" item is just the one that leads to the full set, but we pick one by one.
    # If we can take all, any item is fine, but we should probably pick the one
    # that allows us to take the rest? Actually, if we can take all, the order doesn't matter for total value.
    # But we must pick one. Let's pick the highest density to be safe?
    if np.sum(w_feas) <= remaining_capacity:
        # Pick highest density
        densities_feas = all_densities[feasible_idxs]
        best_local = int(np.argmax(densities_feas))
        return int(feasible_idxs[best_local])
    
    lambda_val = _find_lambda(w_feas, v_feas, remaining_capacity)
    
    # Score candidates
    # Reduced cost: v - lambda * w
    reduced_costs = v_feas - lambda_val * w_feas
    densities_feas = all_densities[feasible_idxs]
    
    # Candidate selection:
    # We want a mix of high reduced cost and high density.
    # Let's take top K by reduced cost and top K by density.
    max_candidates = 30
    if len(feasible_idxs) <= max_candidates:
        candidates = feasible_idxs
    else:
        n_by_rc = max_candidates // 2
        n_by_dens = max_candidates - n_by_rc
        
        # Top by reduced cost
        top_rc_idxs = np.argsort(-reduced_costs)[:n_by_rc]
        top_rc_global = feasible_idxs[top_rc_idxs]
        
        # Top by density
        top_dens_idxs = np.argsort(-densities_feas)[:n_by_dens]
        top_dens_global = feasible_idxs[top_dens_idxs]
        
        # Combine and deduplicate
        candidates = np.unique(np.concatenate([top_rc_global, top_dens_global]))
    
    best_score = -1.0
    best_idx = int(candidates[0])
    
    # Simulate for each candidate
    for idx in candidates:
        # Check time
        if time.time() > _DEADLINE - 5.0:
            break
        
        w_i = w_arr[idx]
        v_i = v_arr[idx]
        rem_cap = remaining_capacity - w_i
        
        if rem_cap < 0:
            continue
        
        total = v_i
        rc = rem_cap
        
        # Greedy fill the rest
        # Iterate through pre-sorted density order
        for j in density_order_all:
            if j == idx:
                continue
            wj = w_arr[j]
            if wj <= rc:
                total += v_arr[j]
                rc -= wj
                if rc <= 0:
                    break
        
        if total > best_score:
            best_score = total
            best_idx = int(idx)
    
    return best_idx
# EVOLVE-BLOCK-END
