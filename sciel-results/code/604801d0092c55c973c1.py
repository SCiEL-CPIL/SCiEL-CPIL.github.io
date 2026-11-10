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
    Returns (maximum_value, list_of_indices_in_optimal_solution).
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, []
    
    # Sort items by density (value/weight) descending
    densities = values / weights
    sorted_indices = np.argsort(-densities)
    
    best_value = 0.0
    best_indices = []
    
    # Stack elements: (next_item_index, current_weight, current_value, current_indices)
    # We store indices to reconstruct the solution.
    # To save memory/time, we can store the path or just backtrack.
    # For N=100, storing a tuple of indices is fine.
    
    # Initial bound check
    if _fractional_bound(weights, values, capacity, 0, sorted_indices) <= best_value:
        return best_value, best_indices

    # Stack for DFS: (next_item_index, current_weight, current_value, current_indices_tuple)
    stack = [(0, 0.0, 0.0, ())]
    
    while stack:
        i, curr_w, curr_v, curr_indices = stack.pop()
        
        # Prune if current weight exceeds capacity
        if curr_w > capacity:
            continue
            
        # Update best value and solution
        if curr_v > best_value:
            best_value = curr_v
            best_indices = list(curr_indices)
            
        # Check if we've processed all items
        if i >= n:
            continue
            
        # Calculate upper bound for this node
        ub = curr_v + _fractional_bound(weights, values, capacity - curr_w, i, sorted_indices)
        
        if ub <= best_value:
            continue
            
        # Branch 1: Exclude item i
        stack.append((i + 1, curr_w, curr_v, curr_indices))
        
        # Branch 2: Include item i
        orig_idx = sorted_indices[i]
        new_w = curr_w + weights[orig_idx]
        if new_w <= capacity:
            new_v = curr_v + values[orig_idx]
            # Add the original index to the solution set
            new_indices = curr_indices + (orig_idx,)
            stack.append((i + 1, new_w, new_v, new_indices))
            
    return best_value, best_indices


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Select the next item using Branch and Bound lookahead on the entire feasible set.
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
    
    if len(feasible_indices) == 0:
        return -1
        
    # Extract weights and values for feasible items
    # We pass these to B&B. B&B will sort them internally.
    # Note: B&B returns indices relative to the input arrays passed to it.
    # So we pass the arrays of feasible items, and get back indices into *those* arrays.
    # We then need to map back to the original `remaining_weights` indices.
    
    sub_weights = remaining_weights[feasible_indices]
    sub_values = remaining_values[feasible_indices]
    
    # Run B&B on the entire feasible subproblem
    # This finds the optimal set of items for the current state.
    opt_val, opt_local_indices = _branch_and_bound(sub_weights, sub_values, remaining_capacity)
    
    if not opt_local_indices:
        return -1
        
    # We have a list of local indices into `sub_weights`/`sub_values` that form an optimal solution.
    # We need to pick one. 
    # Strategy: Pick the one with the highest density from the optimal set to maintain greediness for future steps?
    # Or just pick the first one?
    # Since any item from the optimal set leads to the global optimum (as proven earlier),
    # picking the one with highest density might be slightly better for numerical stability or future pruning,
    # but logically any works. Let's pick the one with highest density in the optimal set.
    
    best_local_idx = -1
    best_density = -1.0
    
    for li in opt_local_indices:
        d = sub_values[li] / sub_weights[li]
        if d > best_density:
            best_density = d
            best_local_idx = li
            
    # Map back to original index
    global_idx = feasible_indices[best_local_idx]
    
    return int(global_idx)

# EVOLVE-BLOCK-END
