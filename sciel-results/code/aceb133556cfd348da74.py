import numpy as np
import time

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Direct heuristic for 0/1 knapsack: pick the item that, when combined with a
    greedy density-based fill of the remaining capacity, yields the highest total value.
    Uses LP relaxation (fractional knapsack) as a smooth surrogate for the integer fill.
    """
    start_time = time.time()
    deadline = 145.0  # 5-second safety margin before 150s kill
    
    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)
    
    n = len(weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Quick check: if no item fits, stop
    if np.all(weights > remaining_capacity):
        return -1
    
    # Filter to only feasible items (weight <= remaining_capacity)
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
    
    # Compute densities (value / weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(f_weights > 0, f_values / f_weights, np.inf)
    
    # Sort by density descending; tie-break by higher value, then lower weight
    sort_keys = np.lexsort((f_weights, -f_values, -densities))
    sorted_f_indices = feasible_indices[sort_keys]
    sorted_weights = f_weights[sort_keys]
    sorted_values = f_values[sort_keys]
    sorted_densities = densities[sort_keys]
    
    # Precompute prefix sums for faster fractional knapsack bound computation
    # prefix_sum_values[i] = sum of values from index 0 to i-1
    # prefix_sum_weights[i] = sum of weights from index 0 to i-1
    prefix_sum_values = np.zeros(f_n + 1)
    prefix_sum_weights = np.zeros(f_n + 1)
    np.cumsum(sorted_values, out=prefix_sum_values[1:])
    np.cumsum(sorted_weights, out=prefix_sum_weights[1:])
    
    # For a small number of feasible items, do a full lookahead
    # For larger, limit candidates to top K
    K = min(f_n, 50)
    
    # Precompute multiple sorting orders for diverse greedy fills
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrt_weights = np.sqrt(f_weights)
        w_075 = f_weights ** 0.75
        mod_density_1 = np.where(sqrt_weights > 0, f_values / sqrt_weights, np.inf)
        mod_density_2 = np.where(w_075 > 0, f_values / w_075, np.inf)
    
    # Strategy 0: density descending (already in sorted_weights/sorted_values order)
    # Strategy 1: value descending
    sort_val = np.lexsort((f_weights, -f_values))
    # Strategy 2: weight ascending
    sort_wt = np.lexsort((-f_values, f_weights))
    # Strategy 3: mod_density_1 descending
    sort_md1 = np.lexsort((f_weights, -f_values, -mod_density_1))
    # Strategy 4: mod_density_2 descending
    sort_md2 = np.lexsort((f_weights, -f_values, -mod_density_2))
    
    # Precompute inverse permutations: inv_X[feasible_idx] = position in X-sorted order
    inv_val = np.zeros(f_n, dtype=np.int64)
    inv_val[sort_val] = np.arange(f_n)
    inv_wt = np.zeros(f_n, dtype=np.int64)
    inv_wt[sort_wt] = np.arange(f_n)
    inv_md1 = np.zeros(f_n, dtype=np.int64)
    inv_md1[sort_md1] = np.arange(f_n)
    inv_md2 = np.zeros(f_n, dtype=np.int64)
    inv_md2[sort_md2] = np.arange(f_n)
    
    # Precompute arrays in each strategy's order
    val_order_w = f_weights[sort_val]
    val_order_v = f_values[sort_val]
    wt_order_w = f_weights[sort_wt]
    wt_order_v = f_values[sort_wt]
    md1_order_w = f_weights[sort_md1]
    md1_order_v = f_values[sort_md1]
    md2_order_w = f_weights[sort_md2]
    md2_order_v = f_values[sort_md2]
    
    def _greedy_fill(w_arr, v_arr, cap, skip_idx):
        """Compute fractional knapsack value filling cap, skipping one index."""
        if cap <= 0:
            return 0.0
        fill_val = 0.0
        rem_cap = cap
        for j in range(len(w_arr)):
            if j == skip_idx:
                continue
            if rem_cap <= 0:
                break
            if w_arr[j] <= rem_cap:
                rem_cap -= w_arr[j]
                fill_val += v_arr[j]
            else:
                if w_arr[j] > 0:
                    fill_val += v_arr[j] * (rem_cap / w_arr[j])
                rem_cap = 0
                break
        return fill_val
    
    best_total_value = -1
    best_global_idx = -1
    
    for k in range(K):
        # Check deadline
        if time.time() - start_time > deadline - 5.0:
            break
        
        i = k
        if sorted_weights[i] > remaining_capacity:
            continue
        
        # Simulate picking item i (index i in the density-sorted feasible array)
        curr_cap = remaining_capacity - sorted_weights[i]
        total_val = sorted_values[i]
        
        # Compute fractional fill using 5 diverse strategies, take max
        frac_0 = _greedy_fill(sorted_weights, sorted_values, curr_cap, i)
        frac_1 = _greedy_fill(val_order_w, val_order_v, curr_cap, inv_val[i])
        frac_2 = _greedy_fill(wt_order_w, wt_order_v, curr_cap, inv_wt[i])
        frac_3 = _greedy_fill(md1_order_w, md1_order_v, curr_cap, inv_md1[i])
        frac_4 = _greedy_fill(md2_order_w, md2_order_v, curr_cap, inv_md2[i])
        
        frac_val = max(frac_0, frac_1, frac_2, frac_3, frac_4)
        total_score = total_val + frac_val
        
        if total_score > best_total_value:
            best_total_value = total_score
            best_global_idx = int(sorted_f_indices[i])
    
    if best_global_idx == -1:
        return -1
    
    return best_global_idx
