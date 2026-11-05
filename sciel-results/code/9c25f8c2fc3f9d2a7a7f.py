import numpy as np
import time

# Global deadline guard with safety margin
_start_time = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin for 120s limit

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add to the knapsack.
    Uses a hybrid approach:
    1. Try exact DP on the remaining subproblem if feasible (size-wise and time-wise).
    2. Fall back to a 1-step lookahead with greedy fill if DP is not feasible.
    
    Returns: index into remaining_weights/values arrays, or -1 to stop.
    """
    if time.time() - _start_time > TIME_LIMIT:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Find indices of items that fit in the remaining capacity
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # If only one feasible item, pick it
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])
    
    # Handle zero-weight items: take the one with highest value
    zero_mask = weights == 0
    if np.any(zero_mask):
        zero_indices = np.where(zero_mask)[0]
        best_zero = zero_indices[np.argmax(values[zero_mask])]
        return int(best_zero)
    
    # --- Try exact DP on the remaining subproblem ---
    # Scale to integers for DP
    scale = 1000
    C_int = int(round(remaining_capacity * scale))
    
    if C_int > 0 and C_int <= 50000:
        # Check if DP is feasible size-wise
        f_n = len(feasible_indices)
        # Only consider feasible items for DP (infeasible ones can't be picked)
        f_weights = weights[feasible_indices]
        f_values = values[feasible_indices]
        
        f_weights_int = np.round(f_weights * scale).astype(int)
        
        # Memory check: if too large, fall back to heuristic
        total_entries = (f_n + 1) * (C_int + 1)
        
        if total_entries <= 10_000_000:
            call_start = time.time()
            
            # DP table: dp[j] = max value with capacity j (1D DP for memory efficiency)
            dp = np.zeros(C_int + 1, dtype=float)
            
            for i in range(f_n):
                w = int(f_weights_int[i])
                v = float(f_values[i])
                
                if w <= C_int:
                    # Update in reverse to avoid using item twice
                    # dp[j] = max(dp[j], dp[j-w] + v)
                    dp[w:] = np.maximum(dp[w:], dp[:C_int+1-w] + v)
                
                # Time guard per item
                if time.time() - call_start > 0.1:
                    # Check global deadline
                    if time.time() - _start_time > TIME_LIMIT:
                        return -1
                    # Continue but check more frequently
                    if (i + 1) % 5 == 0 and time.time() - call_start > 0.5:
                        break
            
            # Backtrack to find one item in the optimal solution
            # We need to track which items are used. Let's use a 2D approach for backtracking
            # or just find the item that contributes most to the solution.
            
            # Actually, 1D DP doesn't support backtracking easily.
            # Let's use a different approach: find the item that is part of an optimal solution.
            # We can do this by checking which items are "critical".
            
            # Alternative: Use 2D DP if memory allows, otherwise use heuristic.
            # For now, let's use a simpler backtracking approach.
            
            # Re-do with 2D DP for backtracking if memory allows
            if total_entries <= 5_000_000:
                call_start2 = time.time()
                dp2d = np.zeros((f_n + 1, C_int + 1), dtype=float)
                
                for i in range(1, f_n + 1):
                    w = int(f_weights_int[i-1])
                    v = float(f_values[i-1])
                    dp2d[i, :] = dp2d[i-1, :]
                    
                    if w <= C_int:
                        dp2d[i, w:] = np.maximum(dp2d[i, w:], dp2d[i-1, :C_int+1-w] + v)
                    
                    if time.time() - call_start2 > 0.1:
                        if time.time() - _start_time > TIME_LIMIT:
                            return -1
                        if (i + 1) % 5 == 0 and time.time() - call_start2 > 0.5:
                            break
                
                # Backtrack
                j = C_int
                for i in range(f_n, 0, -1):
                    if dp2d[i, j] != dp2d[i-1, j]:
                        # Item i-1 is part of the optimal solution
                        return int(feasible_indices[i-1])
                
                # If no item found, fall through to heuristic
            else:
                # 1D DP was used, but we can't backtrack easily.
                # Use heuristic fallback.
                pass
    
    # --- Fallback: 1-step lookahead with greedy fill ---
    # Calculate densities for sorting
    safe_weights = np.where(weights == 0, 1e-9, weights)
    densities = values / safe_weights
    
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    best_total_value = -1
    best_item_index = -1
    
    # For each feasible candidate item, simulate picking it and then filling greedily
    for i in feasible_indices:
        # Value if we pick item i
        current_value = values[i]
        remaining_cap = remaining_capacity - weights[i]
        
        # Fill the rest greedily from the sorted list
        # Skip item i
        for j in sorted_indices:
            if j == i:
                continue
            if weights[j] <= remaining_cap:
                current_value += values[j]
                remaining_cap -= weights[j]
        
        if current_value > best_total_value:
            best_total_value = current_value
            best_item_index = i
    
    return int(best_item_index)
