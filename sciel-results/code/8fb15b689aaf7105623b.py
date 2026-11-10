import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    start_time = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0
    
    # Filter to feasible items only
    feasible_indices = []
    feasible_weights = []
    feasible_values = []
    for i in range(n):
        w = remaining_weights[i]
        v = remaining_values[i]
        if w > 0 and w <= remaining_capacity:
            feasible_indices.append(i)
            feasible_weights.append(w)
            feasible_values.append(v)
        elif w == 0 and v > 0:
            return i  # Free value, take it
            
    if not feasible_indices:
        return -1
        
    m = len(feasible_indices)
    
    # Try DP approach
    # Scale by 1000 to handle 3 decimal places
    SCALE = 1000
    cap_scaled = int(round(remaining_capacity * SCALE))
    
    # Check if capacity is reasonable for DP
    if cap_scaled > 50000: # Safety check, though 12.5 * 1000 = 12500
        # Fallback to greedy + local search
        return _greedy_local_search(feasible_indices, feasible_weights, feasible_values, remaining_capacity, start_time, DEADLINE, SAFETY_MARGIN)
        
    # DP
    # dp[w] = max value
    dp = [0.0] * (cap_scaled + 1)
    # Track items: parent[w] = (prev_w, item_idx_in_feasible)
    # Or simpler: reconstruct by checking items again
    # To reconstruct efficiently, we can store the item that caused the update
    # parent[w] = index in feasible list of the item added to reach state w
    # But we need to know the previous weight too.
    # Standard reconstruction:
    # We can just store the full set of items for each weight? Too much memory.
    # Better: store `take[w]` = item index if item was taken to reach optimal value at weight w.
    # Then to reconstruct:
    # w = cap_scaled
    # while w > 0:
    #   if take[w] != -1:
    #      item = take[w]
    #      w = w - weight[item]
    #   else:
    #      break # This doesn't work directly because we don't track the exact path of weights if we only store max value.
    
    # Correct reconstruction method:
    # We need to know for each weight w, which item was added to get the optimal value.
    # Let's use a 2D array or a list of sets? No.
    # Let's use the standard backtracking with a 2D table or careful 1D backtracking.
    # 1D DP backtracking is tricky because we overwrite states.
    # We can store `last_item[w]` and `prev_w[w]`?
    # Actually, since N is small (100) and C is small (12500), we can use a 2D boolean table `used[i][w]`?
    # 100 * 12500 = 1.25M booleans. This is fine memory-wise.
    
    # Let's use a 2D table for reconstruction: used[i][w] = True if item i is in the optimal solution for capacity w.
    # But we only need the solution for the *full* capacity.
    
    # Alternative: Just run DP, get max value. Then for each item, check if it's in the optimal set by re-running DP? Too slow.
    
    # Let's stick to the `take` array approach with careful backtracking.
    # We will store `take[w]` = index of item in `feasible` list that was added to reach the optimal value at weight `w`.
    # And `prev_weight[w]` = weight before adding that item.
    
    take = [-1] * (cap_scaled + 1)
    prev_weight = [0] * (cap_scaled + 1)
    
    for i in range(m):
        w_i = int(round(feasible_weights[i] * SCALE))
        v_i = feasible_values[i]
        if w_i > cap_scaled:
            continue
        # Iterate backwards to avoid using the same item twice
        for w in range(cap_scaled, w_i - 1, -1):
            # If including item i gives better value than not including it
            if dp[w - w_i] + v_i > dp[w]:
                dp[w] = dp[w - w_i] + v_i
                take[w] = i
                prev_weight[w] = w - w_i
    
    # Reconstruct the set of items in the optimal solution for capacity cap_scaled
    optimal_items = []
    current_w = cap_scaled
    while current_w > 0:
        idx = take[current_w]
        if idx == -1:
            break
        optimal_items.append(idx)
        current_w = prev_weight[current_w]
        
    if not optimal_items:
        # If DP didn't find any items (shouldn't happen if feasible exists and cap > 0), fallback
        # Or if the optimal value is 0
        if dp[cap_scaled] <= 0:
            return -1
        # Fallback to best density item
        best_idx = -1
        best_density = -1
        for i in range(m):
            if feasible_weights[i] > 0:
                density = feasible_values[i] / feasible_weights[i]
                if density > best_density:
                    best_density = density
                    best_idx = i
        if best_idx == -1:
            return -1
        return feasible_indices[best_idx]
        
    # Choose the best item from the optimal set
    # Heuristic: Pick the one with highest density to ensure we make progress
    # or the one that is "most critical"
    best_opt_idx = -1
    best_density = -1
    for i in optimal_items:
        w = feasible_weights[i]
        v = feasible_values[i]
        if w > 0:
            density = v / w
        else:
            density = float('inf')
        if density > best_density:
            best_density = density
            best_opt_idx = i
            
    if best_opt_idx == -1:
        # Should not happen
        return feasible_indices[0]
        
    return feasible_indices[best_opt_idx]

def _greedy_local_search(feasible_indices, feasible_weights, feasible_values, remaining_capacity, start_time, DEADLINE, SAFETY_MARGIN):
    # Fallback logic from previous version, simplified
    m = len(feasible_indices)
    if m == 0:
        return -1
        
    # Sort by density
    items = []
    for i in range(m):
        items.append((feasible_values[i] / feasible_weights[i], feasible_weights[i], feasible_values[i], i))
    items.sort(reverse=True)
    
    best_val = 0
    best_idx = -1
    rem_cap = remaining_capacity
    chosen = []
    
    for density, w, v, idx in items:
        if w <= rem_cap:
            chosen.append(idx)
            rem_cap -= w
            best_val += v
            
    if not chosen:
        return -1
        
    # Simple local search: try swapping
    # This is a fallback, so keep it fast
    current_solution = set(chosen)
    current_value = best_val
    improved = True
    max_swaps = 20
    
    while improved and max_swaps > 0:
        improved = False
        max_swaps -= 1
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        # Try removing each item
        for idx_out in list(current_solution):
            w_out = feasible_weights[idx_out]
            v_out = feasible_values[idx_out]
            new_cap = remaining_capacity - (sum(feasible_weights[i] for i in current_solution) - w_out)
            
            # Try adding each non-selected item
            for idx_in in range(m):
                if idx_in in current_solution:
                    continue
                w_in = feasible_weights[idx_in]
                v_in = feasible_values[idx_in]
                if w_in <= new_cap:
                    new_value = current_value - v_out + v_in
                    if new_value > current_value + 1e-9:
                        current_solution.remove(idx_out)
                        current_solution.add(idx_in)
                        current_value = new_value
                        improved = True
                        break
                        
    # Pick the highest density item from the improved solution
    if not current_solution:
        return -1
        
    best_idx = -1
    best_density = -1
    for idx in current_solution:
        w = feasible_weights[idx]
        v = feasible_values[idx]
        density = v / w if w > 0 else float('inf')
        if density > best_density:
            best_density = density
            best_idx = idx
            
    return feasible_indices[best_idx]
