# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 115.0  # Safety margin before 120s limit

def _fractional_bound_excluding(capacity, weights, values, exclude_indices):
    """
    Calculate the fractional knapsack upper bound for the remaining items,
    excluding items at indices in exclude_indices.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    # Build list of valid items
    items = []
    for i in range(n):
        if i in exclude_indices:
            continue
        w = weights[i]
        v = values[i]
        if w <= 0:
            continue
        items.append((v / w, w, v))
        
    if not items:
        return 0.0
        
    # Sort by density descending
    items.sort(key=lambda x: x[0], reverse=True)
    
    total_val = 0.0
    remaining_cap = capacity
    
    for density, w, v in items:
        if remaining_cap <= 0:
            break
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            # Take fraction
            total_val += v * (remaining_cap / w)
            remaining_cap = 0
            break
            
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item to add to the knapsack using a 2-step lookahead
    heuristic with fractional knapsack upper bound estimation.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    # Find all feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # Step 1: Calculate 1-step scores for all feasible items
    # Score = v_i + fractional_bound(cap - w_i, weights, values, {i})
    scores = {}
    top_candidates = []
    K = 5 # Look at top 5 candidates for 2-step search
    
    for idx in feasible_indices:
        w_i = remaining_weights[idx]
        v_i = remaining_values[idx]
        
        new_cap = remaining_capacity - w_i
        # Use set for exclusion
        completion_val = _fractional_bound_excluding(new_cap, remaining_weights, remaining_values, {idx})
        total_est_val = v_i + completion_val
        
        scores[idx] = total_est_val
        
        # Maintain top K
        if len(top_candidates) < K:
            top_candidates.append((total_est_val, idx))
            top_candidates.sort(key=lambda x: x[0], reverse=True)
        else:
            # If current score is better than the worst in top K, replace
            if total_est_val > top_candidates[-1][0]:
                top_candidates[-1] = (total_est_val, idx)
                top_candidates.sort(key=lambda x: x[0], reverse=True)

    if not top_candidates:
        return -1
        
    # If we have very little time, just return the best 1-step
    if time.monotonic() - _START_TIME > _TIME_LIMIT - 5.0:
        return int(top_candidates[0][1])

    # Step 2: Deep search on top K candidates
    # For each candidate in top K, pick the best second item
    best_total_score = -1.0
    final_best_idx = -1
    
    for score_1, idx_1 in top_candidates:
        w_1 = remaining_weights[idx_1]
        v_1 = remaining_values[idx_1]
        cap_1 = remaining_capacity - w_1
        
        # Find best 2nd item from remaining (excluding idx_1)
        # We need to find the best j != idx_1 such that w_j <= cap_1
        # and v_j + fractional_bound(cap_1 - w_j, ..., {idx_1, j}) is max
        
        best_2nd_total = -1.0
        
        # Iterate through feasible items again, skipping idx_1
        # Optimization: Only consider items that fit in cap_1
        # We can reuse feasible_indices but check weight <= cap_1 and idx != idx_1
        
        for idx_2 in feasible_indices:
            if idx_2 == idx_1:
                continue
            w_2 = remaining_weights[idx_2]
            if w_2 > cap_1:
                continue
                
            v_2 = remaining_values[idx_2]
            cap_2 = cap_1 - w_2
            
            # Calculate bound for the rest excluding idx_1 and idx_2
            completion_val_2 = _fractional_bound_excluding(cap_2, remaining_weights, remaining_values, {idx_1, idx_2})
            
            total_candidate_val = v_1 + v_2 + completion_val_2
            
            if total_candidate_val > best_2nd_total:
                best_2nd_total = total_candidate_val
                
        # If no valid 2nd item was found, best_2nd_total remains -1.
        # In that case, the value is just v_1 (which is covered by score_1 roughly, but score_1 had a fractional bound for the rest).
        # If no item fits, the fractional bound for cap_1 would be 0? No, if no item fits, the loop in _fractional_bound_excluding adds nothing.
        # So score_1 = v_1.
        # If best_2nd_total is -1, it means we didn't find a valid pair. We should compare against the 1-step score?
        # Actually, if no item fits in cap_1, then we can't pick a 2nd item. The total value is v_1.
        # The 1-step score was v_1 + bound(cap_1, {idx_1}). If no item fits, bound is 0. So score_1 = v_1.
        # So if best_2nd_total is -1, we can just use score_1 as the representative value for this idx_1 in the comparison.
        
        effective_score = best_2nd_total if best_2nd_total > -1.0 else score_1
        
        if effective_score > best_total_score:
            best_total_score = effective_score
            final_best_idx = idx_1
            
    # If final_best_idx is still -1, fall back to best 1-step
    if final_best_idx == -1:
        return int(top_candidates[0][1])
        
    return int(final_best_idx)
# EVOLVE-BLOCK-END
