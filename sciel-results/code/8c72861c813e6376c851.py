"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # safety margin before 120s kill


def _fractional_bound(weights, values, capacity, start_idx, sorted_indices):
    """
    Calculate the fractional knapsack upper bound starting from a specific index
    in a pre-sorted list of items (by density).
    """
    if capacity <= 0:
        return 0.0
    
    total_val = 0.0
    remaining_cap = capacity
    
    # Iterate through items in density order
    for i in range(start_idx, len(sorted_indices)):
        idx = sorted_indices[i]
        w = weights[idx]
        v = values[idx]
        
        if w <= remaining_cap:
            total_val += v
            remaining_cap -= w
        else:
            # Fractional part
            total_val += (remaining_cap / w) * v
            break
            
    return total_val


def _branch_and_bound(weights, values, capacity):
    """
    Solve the 0/1 knapsack problem exactly using Branch and Bound.
    Returns the maximum value achievable.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Sort items by density (value/weight) descending
    densities = values / weights
    sorted_indices = np.argsort(-densities)
    
    # Pre-sort weights and values to avoid indexing overhead in the loop?
    # Actually, using indices is fine for N=100.
    
    best_value = 0.0
    # Stack elements: (index, current_weight, current_value)
    # We start by considering the item at index 0 in the sorted list.
    # We use a DFS approach.
    
    # Optimization: If the fractional bound of the entire set is <= best_value, stop.
    # Initial bound:
    if _fractional_bound(weights, values, capacity, 0, sorted_indices) <= best_value:
        return best_value

    # Stack for DFS: (next_item_index, current_weight, current_value)
    stack = [(0, 0.0, 0.0)]
    
    while stack:
        i, curr_w, curr_v = stack.pop()
        
        # Prune if current weight exceeds capacity (shouldn't happen if we check before push)
        if curr_w > capacity:
            continue
            
        # Update best value
        if curr_v > best_value:
            best_value = curr_v
            
        # Check if we've processed all items
        if i >= n:
            continue
            
        # Calculate upper bound for this node
        # The bound assumes we can take fractional items from i onwards
        ub = curr_v + _fractional_bound(weights, values, capacity - curr_w, i, sorted_indices)
        
        if ub <= best_value:
            continue
            
        # Branch 1: Exclude item i
        # Push exclusion branch first so it's processed later (LIFO), or include?
        # Standard B&B often pushes the 'include' branch last to explore it first (depth-first).
        # Let's push 'exclude' first, then 'include' so 'include' is popped next.
        stack.append((i + 1, curr_w, curr_v))
        
        # Branch 2: Include item i
        new_w = curr_w + weights[sorted_indices[i]]
        if new_w <= capacity:
            new_v = curr_v + values[sorted_indices[i]]
            stack.append((i + 1, new_w, new_v))
            
    return best_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item using Branch and Bound lookahead.
    Returns index into remaining arrays, or -1 to stop.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # If the number of feasible items is small, we can just solve the whole thing
    # and pick the first item from the optimal solution?
    # Or just run B&B on the whole set and return one of the items in the optimal set.
    # However, we must return an index into the CURRENT arrays.
    
    # Strategy:
    # 1. Identify top candidates based on density.
    # 2. For each candidate, estimate the total value if we pick it and then solve the rest optimally (or near-optimally).
    # 3. Pick the candidate with the highest estimated total value.
    
    # Sort feasible items by density
    feasible_weights = remaining_weights[feasible_indices]
    feasible_values = remaining_values[feasible_indices]
    
    if len(feasible_indices) == 0:
        return -1
        
    densities = feasible_values / feasible_weights
    sorted_local = np.argsort(-densities)
    
    # We consider the top K candidates for the first pick
    # K=10 is a good balance. B&B on N=100 is fast.
    K = min(10, len(feasible_indices))
    top_k_local = sorted_local[:K]
    
    best_total_value = -1
    best_global_idx = -1
    
    # Pre-calculate some info if needed, but B&B is fast enough for N=100.
    for local_idx in top_k_local:
        global_idx = feasible_indices[local_idx]
        w_pick = remaining_weights[global_idx]
        v_pick = remaining_values[global_idx]
        
        rem_cap = remaining_capacity - w_pick
        if rem_cap < 0:
            continue
            
        # Prepare arrays for the sub-problem (excluding the picked item)
        # We create a mask of items that are feasible and not the picked one
        sub_mask = feasible_mask.copy()
        sub_mask[global_idx] = False
        
        # Get indices of sub-problem items
        sub_indices = np.where(sub_mask)[0]
        
        if len(sub_indices) == 0:
            total_val = v_pick
        else:
            sub_weights = remaining_weights[sub_indices]
            sub_values = remaining_values[sub_indices]
            
            # Solve the sub-problem using B&B
            # If time is running out, we might want to use a faster heuristic
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                # Fallback to greedy for speed
                sub_val = 0.0
                rem_cap_sub = rem_cap
                # Sort sub items by density
                if len(sub_weights) > 0:
                    sub_densities = sub_values / sub_weights
                    sub_sorted = np.argsort(-sub_densities)
                    for si in sub_sorted:
                        if sub_weights[si] <= rem_cap_sub:
                            sub_val += sub_values[si]
                            rem_cap_sub -= sub_weights[si]
                        else:
                            # Try fractional? No, 0/1. Just skip.
                            pass
                total_val = v_pick + sub_val
            else:
                sub_val = _branch_and_bound(sub_weights, sub_values, rem_cap)
                total_val = v_pick + sub_val
                
        if total_val > best_total_value:
            best_total_value = total_val
            best_global_idx = global_idx
            
    return int(best_global_idx)

# EVOLVE-BLOCK-END
