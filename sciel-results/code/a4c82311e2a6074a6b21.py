import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 140.0

def _solve_knapsack_dp(w, v, C):
    """
    Solves 0/1 knapsack exactly using DP.
    Returns a list of indices of the items in the optimal solution.
    """
    n = len(w)
    if n == 0 or C <= 0:
        return []

    # Scale weights to integers to use as array indices
    # Capacity is ~12.5, scale by 100 -> 1250.
    # This precision is sufficient for 100 items.
    scale = 100
    cap_int = int(round(C * scale))
    
    if cap_int < 0:
        return []

    # dp[i][c] = max value using first i items with capacity c
    # We use a 2D array to allow backtracking
    # Dimensions: (n+1) x (cap_int+1)
    # Using float64 for values
    dp = np.zeros((n + 1, cap_int + 1), dtype=np.float64)
    
    # Fill DP table
    for i in range(1, n + 1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        
        # If item weight is 0, it always fits
        # If item weight > cap, it can never fit
        if wi > cap_int:
            # Cannot take item, copy previous row
            dp[i, :] = dp[i-1, :]
        else:
            # Copy previous row as base (not taking item)
            dp[i, :] = dp[i-1, :]
            
            # Update for taking item
            # For c from wi to cap_int:
            # dp[i, c] = max(dp[i-1, c], dp[i-1, c-wi] + vi)
            # Vectorized update
            # dp[i, wi:] = np.maximum(dp[i-1, wi:], dp[i-1, :cap_int-wi+1] + vi)
            
            # Let's do it safely
            prev_row = dp[i-1, :]
            take_values = prev_row[:cap_int - wi + 1] + vi
            dp[i, wi:] = np.maximum(prev_row[wi:], take_values)

    # Backtrack to find selected items
    selected_indices = []
    c = cap_int
    for i in range(n, 0, -1):
        wi = int(round(w[i-1] * scale))
        vi = v[i-1]
        
        # Check if item i-1 was taken
        # If dp[i, c] > dp[i-1, c], it was taken (strictly greater to avoid FP issues, 
        # but with integer weights and float values, we might have ties. 
        # If tie, we prefer not taking? Or taking? 
        # For optimality, either is fine. Let's check if taking leads to the same value.
        # If dp[i, c] == dp[i-1, c], it could be not taken, or taken with 0 net gain (impossible for v>0).
        # If v[i-1] > 0, and it was taken, dp[i, c] should be >= dp[i-1, c].
        # Specifically, if taken, dp[i, c] = dp[i-1, c-wi] + vi.
        # If not taken, dp[i, c] = dp[i-1, c].
        
        # We check if the value at dp[i, c] comes from the "take" branch
        if wi <= c:
            take_val = dp[i-1, c - wi] + vi
            not_take_val = dp[i-1, c]
            
            # Use a small epsilon to handle floating point
            if take_val > not_take_val + 1e-9:
                selected_indices.append(i - 1)
                c -= wi
            else:
                # Not taken (or tie, prefer not taken to save capacity? 
                # Actually, if tie, taking might use capacity for no gain. 
                # Not taking is safer for future steps? 
                # But wait, if we don't take it, we might miss an optimal solution 
                # that *requires* taking it in a different path? 
                # No, DP captures the max value. If values are equal, 
                # the set of optimal solutions includes both. 
                # Picking "not taken" here means we don't include this item.
                # This is a valid choice from the set of optimal solutions.
                pass
        else:
            # Item too heavy, not taken
            pass
            
    selected_indices.reverse()
    return selected_indices

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
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    n = len(remaining_weights)
    if n == 0:
        return -1
        
    w = np.array(remaining_weights, dtype=np.float64)
    v = np.array(remaining_values, dtype=np.float64)
    C = remaining_capacity
    
    # Filter to feasible items first? 
    # The DP handles infeasible items (wi > cap) by just copying the row.
    # So we can pass all items.
    
    # Run DP to get optimal set
    try:
        optimal_indices = _solve_knapsack_dp(w, v, C)
    except Exception:
        # Fallback on error
        feasible = np.where(w <= C)[0]
        if len(feasible) == 0:
            return -1
        densities = v[feasible] / w[feasible]
        densities = np.nan_to_num(densities, nan=0.0, posinf=1e9, neginf=-1e9)
        best_local = np.argmax(densities)
        return int(feasible[best_local])

    if not optimal_indices:
        return -1
        
    # From the optimal set, pick the item with the highest density
    # This helps in case of ties in the DP value, prioritizing efficient items
    sel_w = w[optimal_indices]
    sel_v = v[optimal_indices]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sel_densities = sel_v / sel_w
    sel_densities = np.nan_to_num(sel_densities, nan=0.0, posinf=1e9, neginf=-1e9)
    
    best_pick_rel = np.argmax(sel_densities)
    best_idx = optimal_indices[best_pick_rel]
    
    return int(best_idx)
