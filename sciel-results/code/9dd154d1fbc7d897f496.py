import numpy as np
import time

_start_time = time.time()

def _dp_select_item(w_list, v_list, cap, n, max_time=0.5):
    """
    Use value-scaled DP to find the optimal item to select first.
    Returns the index of the item that should be selected next.
    """
    if n == 0 or cap <= 0:
        return -1
    
    # Check deadline
    if time.time() - _start_time > 110:
        return -1
    
    # Scale values to make DP tractable
    # We want to scale values so that max_total_value is manageable
    total_value = sum(v_list)
    if total_value <= 0:
        return -1
    
    # Choose scaling factor: we want max_scaled_value to be around 5000-10000
    # for reasonable DP size
    target_max_scaled = 5000
    scale = total_value / target_max_scaled
    if scale < 1:
        scale = 1
    
    # Scale values to integers
    scaled_v = [int(v / scale) for v in v_list]
    
    # Ensure at least 1 for non-zero values
    for i in range(n):
        if v_list[i] > 0 and scaled_v[i] == 0:
            scaled_v[i] = 1
    
    max_scaled_v = max(scaled_v) if scaled_v else 0
    if max_scaled_v == 0:
        return -1
    
    # DP: dp[j] = max value achievable with capacity j (in weight units)
    # But we're doing value-based DP: dp[v] = min weight to achieve value v
    # This is more efficient when values are smaller
    
    # Actually, let's do standard weight-based DP
    # dp[w] = max value achievable with weight w
    # But weights can be fractional... let's round them
    
    # Convert weights to integers by scaling
    max_w = max(w_list) if w_list else 1
    if max_w == 0:
        max_w = 1
    
    # Scale weights to integers
    weight_scale = 1000 / max_w if max_w > 0 else 1
    scaled_w = [max(1, int(w * weight_scale)) for w in w_list]
    
    max_scaled_weight = sum(scaled_w)
    if max_scaled_weight > 100000:
        # Too large, fall back to heuristic
        return -1
    
    # DP table: dp[i][j] = max value using first i items with capacity j
    # To save memory, use 1D DP
    dp = [0] * (max_scaled_weight + 1)
    
    # Track which items are used for backtracking
    # parent[i][j] = (prev_j, used_item) or just track the optimal set
    # For simplicity, we'll do a full 2D DP if size is manageable
    
    if max_scaled_weight * n > 1000000:
        # Too large for full DP, use 1D and greedy completion
        # Use 1D DP to compute optimal value, then select best first item
        for i in range(n):
            si = scaled_w[i]
            vi = scaled_v[i]
            for j in range(max_scaled_weight, si - 1, -1):
                if dp[j - si] + vi > dp[j]:
                    dp[j] = dp[j - si] + vi
        
        # Now for each candidate first item, estimate the total value
        # if we pick it first, then fill optimally with remaining
        best_val = -1
        best_idx = -1
        
        for c in range(n):
            wc = scaled_w[c]
            vc = scaled_v[c]
            
            if wc > max_scaled_weight:
                continue
            
            # Value if we take item c first
            # Then we have capacity max_scaled_weight - wc left
            # Estimate remaining value using dp on remaining items
            # For simplicity, use the dp value at capacity max_scaled_weight - wc
            # as an upper bound (it includes all items, so it's optimistic)
            
            # Better: simulate by temporarily removing item c
            # This is expensive, so let's use a simpler estimate
            # The dp value at capacity (max_scaled_weight - wc) gives max value
            # from all items with that capacity. Since we already took item c,
            # the actual remaining value is <= dp[max_scaled_weight - wc]
            
            rem_cap = max_scaled_weight - wc
            if rem_cap < 0:
                rem_cap = 0
            
            est_val = vc + dp[rem_cap]
            
            if est_val > best_val:
                best_val = est_val
                best_idx = c
        
        return best_idx
    
    # Full 2D DP for backtracking
    # dp[i][j] = max value using items 0..i-1 with capacity j
    dp_2d = [[0] * (max_scaled_weight + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        si = scaled_w[i - 1]
        vi = scaled_v[i - 1]
        for j in range(max_scaled_weight + 1):
            # Don't take item i-1
            dp_2d[i][j] = dp_2d[i - 1][j]
            # Take item i-1 if it fits
            if j >= si:
                take_val = dp_2d[i - 1][j - si] + vi
                if take_val > dp_2d[i][j]:
                    dp_2d[i][j] = take_val
    
    # Backtrack to find which items are in the optimal solution
    optimal_items = set()
    j = max_scaled_weight
    for i in range(n, 0, -1):
        if dp_2d[i][j] != dp_2d[i - 1][j]:
            # Item i-1 is in the optimal solution
            optimal_items.add(i - 1)
            j -= scaled_w[i - 1]
    
    if not optimal_items:
        # Fallback to highest value/weight ratio
        best_ratio = -1
        best_idx = -1
        for i in range(n):
            if w_list[i] > 0:
                ratio = v_list[i] / w_list[i]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
        return best_idx
    
    # Select the first item from the optimal set
    # Prefer items with higher value/weight ratio among the optimal set
    best_ratio = -1
    best_idx = -1
    for i in optimal_items:
        if w_list[i] > 0:
            ratio = v_list[i] / w_list[i]
        else:
            ratio = float('inf')
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    
    return best_idx


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Deadline guard
    if time.time() - _start_time > 110:
        # Fallback to simple density
        # Ensure no division by zero
        w = np.asarray(remaining_weights)
        v = np.asarray(remaining_values)
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = v / w
        return int(np.argmax(densities))

    w = remaining_weights
    v = remaining_values
    
    # Convert to lists for faster access in Python loop
    w_list = w.tolist()
    v_list = v.tolist()
    
    # Try DP-based selection first
    dp_result = _dp_select_item(w_list, v_list, remaining_capacity, n)
    
    if dp_result >= 0 and dp_result < n:
        # Verify the item fits
        if w_list[dp_result] <= remaining_capacity:
            return dp_result
    
    # Fallback to original heuristic
    # Calculate densities
    # Handle zero weights just in case
    densities = []
    for i in range(n):
        if w_list[i] == 0:
            densities.append(float('inf'))
        else:
            densities.append(v_list[i] / w_list[i])
            
    # Sort indices by density descending
    indexed_densities = [(d, i) for i, d in enumerate(densities)]
    indexed_densities.sort(key=lambda x: x[0], reverse=True)
    sorted_indices = [i for d, i in indexed_densities]
    
    # Select candidates
    # Top 10 by density
    cand_set = set(sorted_indices[:10])
    # Top 5 by value
    indexed_values = [(v_list[i], i) for i in range(n)]
    indexed_values.sort(key=lambda x: x[0], reverse=True)
    val_indices = [i for val, i in indexed_values[:5]]
    cand_set.update(val_indices)
    
    candidates = list(cand_set)
    
    best_val = -1
    best_idx = -1
    
    for c in candidates:
        wc = w_list[c]
        vc = v_list[c]
        
        cap = remaining_capacity - wc
        if cap < 0:
            continue
            
        val = vc
        
        for idx in sorted_indices:
            if idx == c:
                continue
            wi = w_list[idx]
            vi = v_list[idx]
            
            if wi <= cap:
                val += vi
                cap -= wi
                if cap <= 0:
                    break
            else:
                # Take fraction
                if wi > 0:
                    val += vi * (cap / wi)
                cap = 0
                break
        
        if val > best_val:
            best_val = val
            best_idx = c
            
    if best_idx == -1:
        # Fallback
        # Find max density index from sorted_indices
        return sorted_indices[0]
        
    return best_idx
