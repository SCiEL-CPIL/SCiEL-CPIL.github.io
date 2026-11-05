import numpy as np
import time

# Global timer
_start_time = time.time()
_TIMEOUT = 140.0 # Safety margin before 150s limit

def _solve_knapsack_dp(w, v, C, scale):
    """
    Solves 0/1 knapsack exactly using DP on scaled integer weights.
    Returns (list of indices in optimal solution, total value).
    """
    n = len(w)
    if n == 0:
        return [], 0.0
        
    cap_int = int(round(C * scale))
    if cap_int < 0:
        return [], 0.0

    # dp[i][c] = max value using first i items with capacity c
    # Using numpy for the inner loop might be faster, but standard Python loop is okay for 125k ops.
    # Let's try to optimize the inner loop if possible, but 125k is small.
    
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        # If item weight is 0, it always fits (if value > 0).
        # If wi > cap_int, it can never fit.
        
        # Copy previous row
        # dp[i, :] = dp[i-1, :]
        # Then update for taking item i-1
        
        if wi <= cap_int:
            # For c from wi to cap_int:
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # We can do this with vectorization for speed
            prev_row = dp[i-1, :]
            take_row = prev_row[:cap_int - wi + 1] + vi
            # Compare and take max
            # For c < wi, dp[i, c] = dp[i-1, c]
            # For c >= wi, dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            
            # Initialize with prev row
            dp[i, :] = prev_row
            
            # Update where c >= wi
            # Indices wi to cap_int
            # Compare prev_row[wi:] with take_row
            # take_row corresponds to c=wi...cap_int
            
            # Let's just do a simple loop for clarity and correctness, 125k is fast.
            # But vectorized is better.
            # dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
            
            dp[i, wi:] = np.maximum(prev_row[wi:], take_row)
        else:
            # Item too heavy, copy previous
            dp[i, :] = dp[i-1, :]

    # Backtrack
    selected = []
    c = cap_int
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        # If dp[i, c] != dp[i-1, c], it was taken.
        # Note: Floating point comparison.
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected.append(i - 1)
            wi = int(round(w[i-1] * scale))
            c -= wi
            if c < 0:
                # Should not happen if DP is correct
                break
        # else: not taken

    selected.reverse()
    return selected, dp[n, cap_int]

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights)
        v = np.array(remaining_values)
        feasible = np.where(w <= remaining_capacity)[0]
        if len(feasible) == 0:
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Convert to numpy arrays for speed
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Scale factor
    # Increased from 100 to 1000 to reduce rounding errors in weight discretization
    # This allows the exact DP to more accurately identify items in the true optimal solution set
    scale = 1000
    
    # Check if capacity is reasonable for DP
    cap_int = int(round(C * scale))
    
    # If capacity is too large for DP (memory/time), fallback to heuristic
    if cap_int > 50000: 
        # Fallback to heuristic if capacity is huge
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0: return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    # Run DP
    try:
        selected_indices, total_value = _solve_knapsack_dp(w, v, C, scale)
    except Exception:
        # Fallback on error
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0: return -1
        best_local = np.argmax(densities[feasible])
        return int(feasible[best_local])

    if not selected_indices:
        return -1
        
    # From the optimal set, pick the item with the highest density
    # This ensures we pick an item that is part of the optimal solution
    # and has a good value/weight ratio, which is often a good heuristic for incremental construction
    sel_w = w[selected_indices]
    sel_v = v[selected_indices]
    with np.errstate(divide='ignore', invalid='ignore'):
        sel_densities = sel_v / sel_w
    sel_densities = np.nan_to_num(sel_densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    best_pick_rel = np.argmax(sel_densities)
    best_idx = selected_indices[best_pick_rel]
    
    return int(best_idx)
