import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0

def _greedy_fill(w, v, C, order):
    """
    Perform a greedy fill based on the provided order.
    Returns (indices_list, total_value, total_weight)
    """
    current_indices = []
    current_value = 0.0
    current_weight = 0.0
    
    for idx in order:
        if w[idx] <= C - current_weight:
            current_indices.append(idx)
            current_value += v[idx]
            current_weight += w[idx]
            
    return current_indices, current_value, current_weight

def _local_search_1for1(w, v, C, order, current_indices, current_value, current_weight, max_iters=20):
    """
    Perform 1-for-1 local search on the current solution.
    """
    if not current_indices:
        return current_indices, current_value, current_weight
        
    sol_set = set(current_indices)
    out_set = set(range(len(w))) - sol_set
    
    # Pre-compute densities for sorting outside items
    # We assume 'order' is sorted by density descending.
    # We can just iterate 'order' and check if item is in out_set.
    
    improved = True
    iterations = 0
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        
        # Iterate through outside items in density order
        for i_out in order:
            if i_out in sol_set:
                continue
                
            # Try to swap any item in solution with i_out
            # To be efficient, we can check if adding i_out and removing i_in is beneficial
            # We want: v[i_out] - v[i_in] > 0  AND  w[i_out] - w[i_in] <= slack
            # Or more generally: new_weight <= C and new_value > current_value
            
            # Optimization: Only consider swaps if i_out has higher density or value?
            # Actually, just check all pairs. K is small.
            # We iterate through solution items.
            for i_in in list(current_indices):
                new_weight = current_weight - w[i_in] + w[i_out]
                if new_weight <= C:
                    new_value = current_value - v[i_in] + v[i_out]
                    if new_value > current_value + 1e-9:
                        # Accept swap
                        current_indices.remove(i_in)
                        current_indices.append(i_out)
                        current_weight = new_weight
                        current_value = new_value
                        sol_set.remove(i_in)
                        sol_set.add(i_out)
                        improved = True
                        # Break out of i_in loop, continue checking other i_out?
                        # Usually, we restart or continue. Let's break i_in loop and continue i_out loop.
                        break
                        
    return current_indices, current_value, current_weight

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        if len(remaining_weights) == 0:
            return -1
        # Fallback: Simple greedy
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights)
    v = np.array(remaining_values)
    C = remaining_capacity
    
    # Pre-compute densities and global order
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
    densities[w == 0] = np.inf
    global_order = np.argsort(-densities, kind='stable')
    
    # Identify feasible items (weights <= C)
    # Note: The local search will handle feasibility implicitly by checking new_weight <= C.
    # However, we should only consider items that *can* fit in the current capacity for the initial greedy.
    
    # 1. Initial Greedy Solution
    current_indices, current_value, current_weight = _greedy_fill(w, v, C, global_order)
    
    if not current_indices:
        return -1
        
    # 2. Local Search (1-for-1 swaps)
    # We limit iterations to ensure speed.
    current_indices, current_value, current_weight = _local_search_1for1(
        w, v, C, global_order, current_indices, current_value, current_weight, max_iters=10
    )
    
    # 3. Select the next item
    # We pick the item from the optimized solution that has the highest density.
    # This maintains a greedy-like flow but based on the locally optimal set.
    if not current_indices:
        return -1
        
    # Map local indices back to original indices (they are already 0..n-1)
    # current_indices contains indices into w and v.
    
    # Find the best item in current_indices based on density
    best_idx = -1
    max_density = -np.inf
    for idx in current_indices:
        if densities[idx] > max_density:
            max_density = densities[idx]
            best_idx = idx
            
    return int(best_idx)
