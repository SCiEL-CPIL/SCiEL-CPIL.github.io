import numpy as np
import time

# Global timer to enforce the 120s limit across all calls
_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a DP-based lookahead.
    Returns the index into remaining_weights/remaining_values, or -1 to stop.
    """
    global _start_time
    
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Fallback: Simple greedy by density
        if len(remaining_weights) == 0:
            return -1
        w = np.array(remaining_weights, dtype=np.float64)
        v = np.array(remaining_values, dtype=np.float64)
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
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter feasible items
    feasible_mask = w <= C
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # If the number of feasible items is small, we can afford higher precision or just pick best
    # But let's try DP for all.
    
    # To speed up DP, we can reduce the number of items if many are clearly inferior?
    # No, let's stick to the full set for correctness of the "optimal subproblem" logic.
    
    # Scale weights to integers
    # Capacity is 12.5. Weights are likely around 0.1 - 1.0?
    # Let's use a scale of 1000 to be safe, but check if it's too slow.
    # If N is small, we can use higher scale.
    scale = 1000
    cap_int = int(round(C * scale))
    
    if cap_int <= 0:
        # Capacity too small, pick the lightest feasible item? Or stop?
        # If we can't fit anything, return -1.
        # But we already checked feasible_mask, so at least one item fits.
        # If cap_int is 0 due to rounding, but C > 0, we might have a precision issue.
        # Let's use a slightly higher scale if cap_int is small.
        if C > 0:
            scale = 10000
            cap_int = int(round(C * scale))
        else:
            return -1

    # Prepare integer weights for feasible items only? 
    # No, DP needs to consider all items, but infeasible ones are just skipped.
    # However, for backtracking, it's easier to work with the subset of feasible items 
    # if we map indices back.
    
    # Let's extract feasible items to reduce N in DP if many are infeasible.
    w_feas = w[feasible_mask]
    v_feas = v[feasible_mask]
    n_feas = len(w_feas)
    
    if n_feas == 0:
        return -1
        
    w_int = np.round(w_feas * scale).astype(np.int64)
    
    # Check if any weight is 0
    # If weight is 0 and value > 0, it's always good to take.
    # DP handles this naturally.
    
    # DP Table: dp[i][c] = max value using first i items with capacity c
    # We only need to backtrack, so we store the table.
    # To save memory/time, we can use a 2D array.
    # n_feas is at most 100. cap_int is at most 125000 (if scale=10000).
    # 100 * 125000 = 12.5 million entries. 
    # Float64: 100 MB. This is too much memory for repeated calls?
    # Let's reduce scale.
    # If weights are floats, maybe they have 2 decimal places?
    # Let's try scale = 100. cap_int = 1250.
    # 100 * 1250 = 125,000 entries. 1 MB. Very fast.
    
    scale = 100
    cap_int = int(round(C * scale))
    if cap_int < 1:
        # If capacity is very small, just pick the lightest feasible item
        lightest_idx = np.argmin(w_feas)
        return int(feasible_indices[lightest_idx])

    w_int = np.round(w_feas * scale).astype(np.int64)
    
    # Ensure no negative weights
    w_int = np.maximum(w_int, 0)
    
    # DP
    # dp[i] is an array of length cap_int+1
    # We need the full 2D table for backtracking.
    # dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    # Optimization: Use 1D DP but store the "take" decisions?
    # Or just use 2D. It's small enough.
    
    dp = np.zeros((n_feas + 1, cap_int + 1), dtype=np.float64)
    
    for i in range(1, n_feas + 1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Copy previous row
        dp[i, :] = dp[i-1, :]
        
        if wi <= cap_int:
            # Update
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # Vectorized
            take_vals = dp[i-1, :cap_int+1-wi] + vi
            prev_vals = dp[i-1, wi:cap_int+1]
            
            # We want to update dp[i, wi:cap_int+1]
            # max(prev_vals, take_vals)
            # Note: take_vals has length cap_int+1-wi
            # prev_vals has length cap_int+1-wi
            
            updated = np.maximum(prev_vals, take_vals)
            dp[i, wi:cap_int+1] = updated
        # else: item too heavy, row is already copied
    
    # Backtrack to find one item in the optimal solution
    # We traverse from the last item to the first.
    # The first item we find that was "taken" is a valid choice.
    # Is it the *best* choice? 
    # It is *an* item in *an* optimal solution. 
    # Picking any item from an optimal solution guarantees that the remaining subproblem 
    # can still be solved optimally (optimal substructure).
    
    c = cap_int
    selected_idx = -1
    
    for i in range(n_feas, 0, -1):
        wi = w_int[i-1]
        vi = v_feas[i-1]
        
        # Check if item i-1 was taken
        # If dp[i, c] > dp[i-1, c], it was taken.
        # We use a small epsilon for float comparison.
        if dp[i, c] > dp[i-1, c] + 1e-9:
            selected_idx = i - 1
            c -= wi
            if c < 0:
                break
            break
        
    if selected_idx != -1:
        # Map back to original index
        return int(feasible_indices[selected_idx])
    else:
        # If no item was taken in the backtrack (shouldn't happen if optimal value > 0),
        # fallback to best density
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v_feas / w_feas
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible_indices[best_local])
