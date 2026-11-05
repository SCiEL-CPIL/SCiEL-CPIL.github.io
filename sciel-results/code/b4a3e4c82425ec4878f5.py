import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 145.0

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    if time.time() - _start_time > _TIMEOUT:
        return -1

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    if C < 0:
        return -1
        
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    n_feas = len(feasible_indices)
    
    # Handle zero-weight items: pick them all if they have positive value
    # If there are zero-weight positive value items, we should pick one.
    # Which one? Any. But picking the one with highest value might be better?
    # Actually, since they don't consume capacity, we should pick ALL of them eventually.
    # The heuristic picks one now. The next call will pick the next one.
    # So picking any is fine, but picking the highest value one first doesn't hurt.
    zero_w_pos_v = (w == 0) & (v > 0)
    if np.any(zero_w_pos_v):
        # Pick the zero-weight item with the highest value
        zw_indices = np.where(zero_w_pos_v)[0]
        zw_values = v[zw_indices]
        best_zw_local_idx = np.argmax(zw_values)
        return int(zw_indices[best_zw_local_idx])
    
    # For positive weight feasible items
    pos_w = w > 0
    # Densities
    densities = np.zeros(n)
    densities[pos_w] = v[pos_w] / w[pos_w]
    
    # Get feasible items
    feas_w = w[feasible_indices]
    feas_v = v[feasible_indices]
    feas_dens = densities[feasible_indices]
    
    # Sort by density descending
    sort_order = np.argsort(-feas_dens, kind='stable')
    sorted_feas_w = feas_w[sort_order]
    sorted_feas_v = feas_v[sort_order]
    sorted_feas_idx = feasible_indices[sort_order]
    
    # Precompute prefix sums
    # prefix_w[i] = sum of sorted_feas_w[0..i-1]
    # prefix_v[i] = sum of sorted_feas_v[0..i-1]
    prefix_w = np.zeros(n_feas + 1)
    prefix_v = np.zeros(n_feas + 1)
    # Use cumsum for speed
    prefix_w[1:] = np.cumsum(sorted_feas_w)
    prefix_v[1:] = np.cumsum(sorted_feas_v)
    
    best_score = -1e10
    best_idx = -1
    
    # For each candidate in the sorted order
    for pos_c in range(n_feas):
        c = sorted_feas_idx[pos_c]
        w_c = sorted_feas_w[pos_c]
        v_c = sorted_feas_v[pos_c]
        
        rem_cap = C - w_c
        if rem_cap < 0:
            continue
            
        sim_val = v_c
        
        # Block 1: Items 0 to pos_c - 1
        # Find largest j such that prefix_w[j] <= rem_cap
        # j is in [0, pos_c]
        # We can use searchsorted
        # We want the rightmost index j such that prefix_w[j] <= rem_cap
        # prefix_w is sorted ascending
        j = np.searchsorted(prefix_w, rem_cap, side='right')
        # Ensure j does not exceed pos_c
        if j > pos_c:
            j = pos_c
            
        # Take items 0 to j-1
        sim_val += prefix_v[j]
        rem_cap -= prefix_w[j]
        
        # Block 2: Items pos_c + 1 to n_feas - 1
        if rem_cap > 0 and pos_c + 1 < n_feas:
            start_idx = pos_c + 1
            # We want to find largest e such that prefix_w[e] - prefix_w[start_idx] <= rem_cap
            # prefix_w[e] <= rem_cap + prefix_w[start_idx]
            target = rem_cap + prefix_w[start_idx]
            # e must be >= start_idx
            # searchsorted in prefix_w
            e = np.searchsorted(prefix_w, target, side='right')
            # Ensure e is not beyond n_feas
            if e > n_feas:
                e = n_feas
            
            # Take items start_idx to e-1
            sim_val += prefix_v[e] - prefix_v[start_idx]
            
        if sim_val > best_score:
            best_score = sim_val
            best_idx = c
            
    if best_idx < 0 or best_score <= 0:
        return -1
        
    return int(best_idx)
