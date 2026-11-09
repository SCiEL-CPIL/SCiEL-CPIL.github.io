import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    w = np.asarray(remaining_weights, dtype=float)
    v = np.asarray(remaining_values, dtype=float)
    C = float(remaining_capacity)
    
    # Precompute densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = v / w
        # If weight is 0, density is inf. 
        # If value is 0, density is 0.
        # If both 0, undefined. Assume 0.
        densities[~np.isfinite(densities)] = 0.0
        
    best_total_val = -1.0
    best_idx = -1
    
    # If only 1 item, pick it
    if n == 1:
        return 0

    # 2-step lookahead
    for i in range(n):
        w_i = w[i]
        v_i = v[i]
        C_rem = C - w_i
        
        if C_rem < 0:
            continue
            
        # Find best j != i with w_j <= C_rem
        # Vectorized mask
        # We can pre-compute sorted indices? No, C_rem changes.
        
        # Mask: j != i AND w_j <= C_rem
        # Create mask for w <= C_rem
        # Then set i to False
        
        # To avoid creating new arrays in loop, we can use indices
        # But n=100 is small.
        
        feasible = (w <= C_rem)
        feasible[i] = False # Exclude i
        
        if not np.any(feasible):
            total_val = v_i
        else:
            # Get densities of feasible items
            d_feasible = densities[feasible]
            # Find max density
            max_d = np.max(d_feasible)
            
            # Among those with max density, find max value
            # This is a bit complex with boolean indexing
            # Alternative: 
            # Create a score array for feasible items: density * 1e10 + value
            # This ensures max density is primary, max value secondary
            # But we only want to consider feasible items.
            
            # Let's just find the indices
            # idx_feasible = np.where(feasible)[0]
            # d_vals = densities[idx_feasible]
            # v_vals = v[idx_feasible]
            
            # Find max density
            # If we just take argmax of densities, we get one index.
            # But we want to handle ties by value.
            
            # Trick: Score = density + epsilon * value
            # If densities are distinct, this works.
            # If densities are equal, higher value wins.
            # What epsilon?
            # Densities can be large. Values can be large.
            # Let's use the tie-breaking logic explicitly.
            
            # Get indices of max density
            # max_d is scalar
            # mask_max_d = (d_feasible == max_d)
            # If we use d_feasible (which is a view/slice), we need to map back.
            
            # Simpler:
            # Create a combined score for all items, but set non-feasible to -inf
            # Score = density * 1e100 + value  (Too big?)
            # Let's use: Score = density + value * 1e-10
            # This might fail if density differences are tiny.
            
            # Robust way:
            # 1. Find max density among feasible
            # 2. Find max value among feasible with that density
            
            # We have d_feasible and feasible mask.
            # Let's get the actual values.
            v_feasible = v[feasible]
            
            # Find indices in v_feasible where d_feasible == max_d
            # Since floating point comparison is risky, use np.isclose?
            # Or just assume densities are distinct enough?
            # In random instances, ties are rare.
            # If we just pick the first max density, it's fine.
            
            # Let's just pick the item with max density. If tie, max value.
            # We can do this by sorting? No, too slow.
            
            # Let's use the score: d + v * 1e-10
            # This is safe if d is not huge.
            # Or just:
            # max_idx_in_feasible = np.argmax(d_feasible + v_feasible * 1e-10)
            # v_best_next = v_feasible[max_idx_in_feasible]
            
            v_best_next = np.max(v_feasible[(d_feasible == max_d)])
            total_val = v_i + v_best_next
            
        if total_val > best_total_val:
            best_total_val = total_val
            best_idx = i

    return int(best_idx)
