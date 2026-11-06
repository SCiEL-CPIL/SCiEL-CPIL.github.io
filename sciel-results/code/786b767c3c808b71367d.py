"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin for 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item for a 0/1 Knapsack problem.
    Uses exact DP with prefix/suffix tables if weights can be scaled to integers.
    Falls back to LP relaxation heuristic otherwise.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Find feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Try to determine if weights are integer-scaled
    # Check scales: 1, 10, 100
    scale = 0
    for s in [1, 10, 100]:
        scaled_weights = weights * s
        # Check if all are close to integers
        if np.allclose(scaled_weights, np.round(scaled_weights), atol=1e-9):
            scale = s
            break
            
    if scale > 0:
        # Integer DP path
        w_int = np.round(weights * scale).astype(np.int64)
        c_int = int(np.round(remaining_capacity * scale))
        
        # If capacity is too large, DP might be slow/memory heavy
        # For c_int=125, N=100, it's fine.
        if c_int > 1000:
             scale = 0 # Fallback
        else:
            # Build prefix DP tables
            # dp_prefix[i] is an array of length c_int + 1
            # dp_prefix[i][j] = max value using items 0..i-1 with capacity j
            dp_prefix = np.zeros((n + 1, c_int + 1), dtype=np.float64)
            for i in range(n):
                wi = w_int[i]
                vi = values[i]
                if wi <= c_int:
                    # dp_prefix[i+1][j] = max(dp_prefix[i][j], dp_prefix[i][j-wi] + vi)
                    # We can do this vectorized
                    dp_prefix[i+1] = np.maximum(dp_prefix[i], dp_prefix[i] - vi) # Placeholder
                    # Actually, simple loop is safer and fast enough for C=125
                    dp_prefix[i+1] = dp_prefix[i].copy()
                    for j in range(c_int, wi - 1, -1):
                        val = dp_prefix[i][j - wi] + vi
                        if val > dp_prefix[i+1][j]:
                            dp_prefix[i+1][j] = val
                else:
                    dp_prefix[i+1] = dp_prefix[i].copy()
            
            # Build suffix DP tables
            # dp_suffix[i] is an array of length c_int + 1
            # dp_suffix[i][j] = max value using items i..n-1 with capacity j
            dp_suffix = np.zeros((n + 1, c_int + 1), dtype=np.float64)
            for i in range(n - 1, -1, -1):
                wi = w_int[i]
                vi = values[i]
                dp_suffix[i] = dp_suffix[i+1].copy()
                if wi <= c_int:
                    for j in range(c_int, wi - 1, -1):
                        val = dp_suffix[i+1][j - wi] + vi
                        if val > dp_suffix[i][j]:
                            dp_suffix[i][j] = val
            
            best_score = -1.0
            best_idx = -1
            
            for i in feasible_indices:
                wi = w_int[i]
                vi = values[i]
                c_rem = c_int - wi
                
                if c_rem < 0:
                    continue
                
                # Calculate max value of rest (items excluding i) with capacity c_rem
                # Combining dp_prefix[i] (items 0..i-1) and dp_suffix[i+1] (items i+1..n-1)
                # We need max_{c=0..c_rem} (dp_prefix[i][c] + dp_suffix[i+1][c_rem - c])
                
                # Vectorized convolution-like max
                # c ranges from 0 to c_rem
                # dp_prefix[i][0..c_rem] and dp_suffix[i+1][c_rem..0]
                p_part = dp_prefix[i][:c_rem+1]
                s_part = dp_suffix[i+1][c_rem::-1]
                
                # We want max of p_part[j] + s_part[j] for j in 0..c_rem
                # This is element-wise sum, then max
                combined = p_part + s_part
                v_rest = np.max(combined)
                
                total_score = vi + v_rest
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = i
            
            if best_idx != -1:
                return int(best_idx)
            else:
                return -1

    # Fallback: LP Relaxation Heuristic
    # Pre-compute density for sorting
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    # Sort indices by density descending
    sorted_indices = np.argsort(densities)[::-1]
    
    best_score = -1.0
    best_idx = -1
    
    # Evaluate each feasible candidate
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        rem_cap = remaining_capacity - w_i
        
        # Calculate LP relaxation value for remaining items in rem_cap
        lp_val = 0.0
        cur_cap = rem_cap
        
        # Iterate through items sorted by density
        for j in sorted_indices:
            if j == i:
                continue
            w_j = weights[j]
            v_j = values[j]
            
            if w_j <= cur_cap:
                lp_val += v_j
                cur_cap -= w_j
                if cur_cap <= 0:
                    break
            else:
                # Fractional part
                if w_j > 0:
                    lp_val += v_j * (cur_cap / w_j)
                cur_cap = 0
                break
        
        total_score = v_i + lp_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
            
    return int(best_idx)
# EVOLVE-BLOCK-END
