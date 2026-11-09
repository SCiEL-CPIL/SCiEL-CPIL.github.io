"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 5.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def _greedy_fill(capacity, weights, values, sorted_indices, excluded_indices):
    """
    Greedily fill the knapsack with items from sorted_indices (by density),
    excluding those in excluded_indices.
    Returns total value.
    """
    total_val = 0.0
    remaining_cap = capacity
    # We use a set for O(1) lookup of excluded indices, though for small K, list is fine
    # But excluded_indices changes per candidate in the outer loop.
    # To speed this up, we can just check `idx not in excluded_set`.
    
    # Optimization: If capacity is very small or no items fit, return 0
    if remaining_cap <= 0:
        return 0.0
        
    for idx in sorted_indices:
        if idx in excluded_indices:
            continue
        w = weights[idx]
        if w <= remaining_cap:
            remaining_cap -= w
            total_val += values[idx]
            if remaining_cap <= 0:
                break
    return total_val

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time budget
    if not _elapsed_ok():
        # Fallback to simple greedy
        densities = remaining_values / remaining_weights
        feasible = remaining_weights <= remaining_capacity
        if not np.any(feasible):
            return -1
        masked_densities = np.where(feasible, densities, -1)
        return int(np.argmax(masked_densities))

    # 1. Compute densities
    densities = remaining_values / remaining_weights
    
    # 2. Sort indices by density descending
    # We only care about items that fit in the current capacity for the first pick
    # But for the fill, items might not fit individually but we check in loop.
    # Actually, for the first pick, it MUST fit.
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    # Get indices of feasible items
    feasible_indices = np.where(feasible_mask)[0]
    
    # Sort feasible indices by density descending
    # We need to sort based on densities[indices]
    sorted_feasible_indices = feasible_indices[np.argsort(-densities[feasible_indices])]
    
    # Limit candidates for step 1 to top K1
    K1 = 20
    candidates_1 = sorted_feasible_indices[:K1]
    
    # Sort ALL indices by density for the greedy fill
    # We need a global sorted order for the fill simulation
    all_indices = np.arange(n)
    sorted_all_indices = all_indices[np.argsort(-densities)]
    
    best_total_value = -1.0
    best_index = -1
    
    # 3. Limited 2-step lookahead
    # For each candidate c1 in candidates_1:
    for c1 in candidates_1:
        w1 = remaining_weights[c1]
        v1 = remaining_values[c1]
        
        # Capacity left after picking c1
        cap_left = remaining_capacity - w1
        
        # Candidates for step 2: Top K2 from sorted_all_indices, excluding c1, and fitting in cap_left
        # We iterate through sorted_all_indices to find top K2 that fit and aren't c1
        candidates_2 = []
        for idx in sorted_all_indices:
            if idx == c1:
                continue
            if remaining_weights[idx] <= cap_left:
                candidates_2.append(idx)
                if len(candidates_2) >= 10: # K2 = 10
                    break
        
        # If no candidates for step 2, just evaluate greedy fill for c1
        if not candidates_2:
            # Greedy fill excluding c1
            excluded = {c1}
            fill_val = _greedy_fill(cap_left, remaining_weights, remaining_values, sorted_all_indices, excluded)
            total_val = v1 + fill_val
            if total_val > best_total_value:
                best_total_value = total_val
                best_index = c1
            continue
            
        # For each candidate c2 in candidates_2:
        for c2 in candidates_2:
            w2 = remaining_weights[c2]
            v2 = remaining_values[c2]
            
            cap_left2 = cap_left - w2
            
            # Greedy fill excluding c1 and c2
            excluded = {c1, c2}
            fill_val = _greedy_fill(cap_left2, remaining_weights, remaining_values, sorted_all_indices, excluded)
            
            total_val = v1 + v2 + fill_val
            
            if total_val > best_total_value:
                best_total_value = total_val
                best_index = c1
                
    # If no best index found (shouldn't happen if feasible_mask had items), fallback
    if best_index == -1:
        best_index = int(candidates_1[0])
        
    return best_index
# EVOLVE-BLOCK-END
