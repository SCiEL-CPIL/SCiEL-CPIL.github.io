"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Filter out items that are individually too heavy?
    # The DP will handle it, but we can speed up by ignoring items with w > C.
    # However, the index returned must correspond to the original array.
    # So we must be careful.
    
    total_value = sum(remaining_values)
    if total_value <= 0:
        return -1
        
    # Determine scale for value DP
    # Target DP size ~ 2000-5000
    target_size = 3000
    scale = target_size / total_value
    # Ensure scale is not too small to cause many collisions, 
    # but not so large that DP is huge.
    # If total_value is very small, scale becomes large.
    # max_v_scaled = int(total_value * scale) ~ target_size
    
    max_v_scaled = int(round(total_value * scale))
    
    # Safety cap on max_v_scaled to prevent memory/time issues if total_value is weird
    if max_v_scaled > 10000:
        max_v_scaled = 10000
        scale = max_v_scaled / total_value
        
    # DP: min_weight[v] = minimum weight required to achieve exactly value v
    # Initialize with infinity
    INF = float('inf')
    dp = [INF] * (max_v_scaled + 1)
    dp[0] = 0.0
    
    # To allow backtracking, we store the previous state's value index
    # prev[i][v] = value index in previous row that led to current v
    # If item i was NOT used, prev[i][v] = v
    # If item i WAS used, prev[i][v] = v - v_i_scaled
    
    # We can optimize memory by only storing the path if we backtrack.
    # But N is small (100). Let's store the parent pointers.
    # prev[i][v] for i in 1..n, v in 0..max_v_scaled
    # This is a list of lists or a numpy array.
    # Numpy is faster for initialization and access if we do it carefully, 
    # but Python lists are okay for 100x3000 = 300k elements.
    
    # Let's use a list of lists for prev
    prev = [[0] * (max_v_scaled + 1) for _ in range(n + 1)]
    
    # Current dp row
    curr_dp = dp[:]
    
    for i in range(n):
        wi = remaining_weights[i]
        vi = remaining_values[i]
        
        # If item is too heavy, it can't be part of any feasible solution 
        # that fits in remaining_capacity, BUT it might be part of the 
        # "min weight to achieve value" calculation if we just want the 
        # optimal set. 
        # Wait, the DP calculates min weight for ANY value.
        # If wi > remaining_capacity, then any solution including it has 
        # weight > remaining_capacity, so it's infeasible for the final selection.
        # We can skip updating DP for this item?
        # No, if we skip it, the DP reflects "min weight using items 0..i-1".
        # If we include it, dp might show a lower weight for some value?
        # No, adding an item increases weight. 
        # `dp_new[v] = min(dp_old[v], dp_old[v-vi] + wi)`.
        # Since wi > 0, `dp_old[v-vi] + wi > dp_old[v-vi]`.
        # It doesn't necessarily make dp_new[v] worse than dp_old[v] if dp_old[v] was INF.
        # But if the final check is `dp[v] <= remaining_capacity`, 
        # and `wi > remaining_capacity`, then any `v` achieved using item `i` 
        # will have weight at least `wi` (plus others), so `> remaining_capacity`.
        # So such items will never be selected in the final backtrack 
        # IF we check feasibility against `remaining_capacity`.
        # So we can safely include them in DP, or skip them.
        # Skipping them saves time.
        if wi > remaining_capacity:
            # Item i is not feasible on its own, so it can't be in the final knapsack.
            # Copy previous state
            for v in range(max_v_scaled + 1):
                prev[i+1][v] = v
            # curr_dp remains unchanged
            continue
            
        vi_scaled = int(round(vi * scale))
        if vi_scaled <= 0:
            # Value is negligible or zero. 
            # If vi > 0 but vi_scaled == 0, we lose precision.
            # We can try to increase scale? 
            # For now, if vi_scaled == 0, we treat it as 0 value.
            # This item adds weight but no value (in scaled terms).
            # It will never be chosen in min-weight DP for a specific value > 0.
            # It might be chosen if we allow 0 value? No, we want max value.
            # So we can skip it or just copy.
            for v in range(max_v_scaled + 1):
                prev[i+1][v] = v
            continue
            
        new_dp = curr_dp[:]
        
        # Update DP
        # Iterate v from max_v_scaled down to vi_scaled
        # We need to update prev[i+1] as well
        # prev[i+1][v] = v (if not used) or v - vi_scaled (if used)
        
        # Optimization: Only iterate up to max_v_scaled
        for v in range(max_v_scaled, vi_scaled - 1, -1):
            prev_val = curr_dp[v - vi_scaled]
            if prev_val != INF:
                new_weight = prev_val + wi
                if new_weight < curr_dp[v]:
                    new_dp[v] = new_weight
                    prev[i+1][v] = v - vi_scaled
                else:
                    prev[i+1][v] = v
            else:
                # Cannot achieve v-vi_scaled, so cannot use item i to reach v
                # Keep old state
                if curr_dp[v] != INF:
                     prev[i+1][v] = v
                else:
                     prev[i+1][v] = v # Doesn't matter, it's INF
        # For v < vi_scaled, item i cannot be used to reach v (since v-vi_scaled < 0)
        for v in range(vi_scaled):
            prev[i+1][v] = v
            
        curr_dp = new_dp

    # Find the best value
    best_v = 0
    for v in range(max_v_scaled, -1, -1):
        if curr_dp[v] <= remaining_capacity:
            best_v = v
            break
            
    if best_v == 0:
        return -1
        
    # Backtrack to find an item
    # Start at i=n, v=best_v
    i = n
    v = best_v
    while i > 0:
        prev_v = prev[i][v]
        if prev_v != v:
            # Item i-1 was used
            return i - 1
        else:
            # Item i-1 was not used
            v = prev_v
            i -= 1
            
    return -1
# EVOLVE-BLOCK-END
