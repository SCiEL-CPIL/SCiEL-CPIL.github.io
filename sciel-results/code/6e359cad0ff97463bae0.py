"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0
_start_time = None


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _fast_greedy_complete(capacity, weights, values):
    """
    Fast greedy completion.
    Returns the total value obtained by greedily filling the knapsack
    with the given items (by density) until capacity is exhausted.
    """
    if capacity <= 0 or len(values) == 0:
        return 0.0
    
    # Filter feasible items
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    
    fw = weights[feasible]
    fv = values[feasible]
    
    if len(fw) == 0:
        return 0.0

    # Calculate densities
    densities = fv / fw
    
    # Sort by density descending
    # Using argsort is O(N log N)
    order = np.argsort(densities)[::-1]
    
    total_value = 0.0
    remaining_cap = capacity
    
    # Iterate through sorted items
    for idx in order:
        if remaining_cap <= 0:
            break
        w = fw[idx]
        v = fv[idx]
        if w <= remaining_cap:
            total_value += v
            remaining_cap -= w
            
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Check deadline
    if _get_elapsed() > _DEADLINE - 2.0:
        # Fallback to simple greedy
        feasible_mask = remaining_weights <= remaining_capacity
        if not np.any(feasible_mask):
            return -1
        feasible_idx = np.where(feasible_mask)[0]
        fw = remaining_weights[feasible_idx]
        fv = remaining_values[feasible_idx]
        densities = fv / fw
        return int(feasible_idx[np.argmax(densities)])

    # Identify feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    feasible_idx = np.where(feasible_mask)[0]
    fw = remaining_weights[feasible_idx]
    fv = remaining_values[feasible_idx]
    fn = len(fw)

    if fn == 1:
        return int(feasible_idx[0])

    # Select top K candidates by density for the first step
    K = min(fn, 12)
    densities = fv / fw
    top_k_local = np.argsort(densities)[::-1][:K]
    
    best_total = -1.0
    best_local = top_k_local[0] # Default to best density

    # 2-Step Lookahead
    # For each first-level candidate i:
    for i in top_k_local:
        if _get_elapsed() > _DEADLINE - 5.0:
            break
            
        w_i = fw[i]
        v_i = fv[i]
        cap_after_i = remaining_capacity - w_i
        
        if cap_after_i <= 0:
            # Only this item fits
            total_val = v_i
            if total_val > best_total:
                best_total = total_val
                best_local = i
            continue
            
        # Identify candidates for the second step from the remaining items
        # Create a mask to exclude item i
        # Note: fw and fv are indexed by local indices 0..fn-1
        # We need to consider all j != i such that fw[j] <= cap_after_i
        
        # To speed up, we can just iterate through all feasible items except i
        # But checking all fn items for each i is K*fn. 12*100 = 1200 checks.
        # Then for the best M second-level candidates, we run greedy.
        
        # Let's find the top M second-level candidates by density among those that fit
        # We can filter first:
        feasible_j_mask = np.ones(fn, dtype=bool)
        feasible_j_mask[i] = False
        feasible_j_mask &= (fw <= cap_after_i)
        
        if not np.any(feasible_j_mask):
            total_val = v_i # No second item fits
            if total_val > best_total:
                best_total = total_val
                best_local = i
            continue
            
        j_idxs = np.where(feasible_j_mask)[0]
        j_densities = fv[j_idxs] / fw[j_idxs]
        
        M = min(len(j_idxs), 8)
        top_m_local = np.argsort(j_densities)[::-1][:M]
        
        # For each second-level candidate j:
        for j_rel in top_m_local:
            j = j_idxs[j_rel]
            w_j = fw[j]
            v_j = fv[j]
            cap_after_j = cap_after_i - w_j
            
            if cap_after_j <= 0:
                total_val = v_i + v_j
            else:
                # Run greedy on the rest
                # Exclude i and j
                rest_mask = np.ones(fn, dtype=bool)
                rest_mask[i] = False
                rest_mask[j] = False
                rest_mask &= (fw <= cap_after_j)
                
                if np.any(rest_mask):
                    rest_w = fw[rest_mask]
                    rest_v = fv[rest_mask]
                    greedy_val = _fast_greedy_complete(cap_after_j, rest_w, rest_v)
                else:
                    greedy_val = 0.0
                total_val = v_i + v_j + greedy_val
            
            if total_val > best_total:
                best_total = total_val
                best_local = i

    return int(feasible_idx[best_local])
# EVOLVE-BLOCK-END
