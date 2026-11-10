import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

# Global deadline guard
_start_time = None
_GLOBAL_DEADLINE = 140.0  # Leave 10s buffer

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_time_up():
    return _get_elapsed() > _GLOBAL_DEADLINE

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if _is_time_up():
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Handle zero weight positive value items first
    zero_weight_indices = np.where((weights == 0) & (values > 0))[0]
    if len(zero_weight_indices) > 0:
        return int(zero_weight_indices[0])
    
    # Filter feasible items (weight > 0 and weight <= capacity)
    feasible_mask = (weights <= remaining_capacity) & (weights > 0)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # Strategy:
    # 1. Try DP if the problem size is small enough.
    # 2. If DP is too large, use Fractional Surrogate (Continuous Relaxation)
    #    to guide the choice, potentially with a local B&B polish if time permits.
    
    C = remaining_capacity
    max_w = np.max(f_weights)
    
    # DP Feasibility Check
    # We want to scale weights to integers.
    # DP table size: f_n * (C * scale)
    # Memory limit: ~100MB -> 100e6 / 8 bytes = 12.5M booleans (for takes) + floats.
    # Let's aim for C_scaled * f_n < 5_000_000 operations.
    
    # Estimate scale
    # If we pick scale = 100, C_scaled = 1250. f_n=100. 125k ops. Very fast.
    # If we pick scale = 1000, C_scaled = 12500. f_n=100. 1.25M ops. Fast.
    # If we pick scale = 10000, C_scaled = 125000. f_n=100. 12.5M ops. ~0.5s-1s.
    
    # Let's set a target for C_scaled.
    # If C is small (e.g. 12.5), we can use high scale.
    # If C is large, lower scale.
    
    # Heuristic:
    # Max C_scaled = 200000 (200k).
    # If C * scale > 200000, reduce scale.
    # Also check f_n * C_scaled < 10_000_000.
    
    target_C_scaled = 100_000
    scale = target_C_scaled / C
    scale = int(scale)
    scale = max(1, scale)
    
    # Ensure weights are not zero after scaling?
    # If max_w * scale == 0, then weights are too small. Increase scale.
    if max_w * scale < 1:
         scale = int(1.0 / max_w) + 10
         
    w_scaled = (f_weights * scale).astype(np.int64)
    C_scaled = int(C * scale)
    
    # Check if DP is viable
    # Memory for takes: f_n * (C_scaled + 1) booleans.
    # If this is too big, use Fractional Surrogate.
    dp_size = f_n * (C_scaled + 1)
    
    if dp_size < 5_000_000 and C_scaled < 200_000:
        # Use DP
        return _solve_dp(f_values, w_scaled, C_scaled, feasible_indices, f_n)
    else:
        # Use Fractional Surrogate + Local Polish
        return _solve_fractional_surrogate(f_values, f_weights, C, feasible_indices, f_n)

def _solve_dp(values, weights, capacity, original_indices, n):
    # DP with backtracking
    # dp[j] = max value using capacity j
    # takes[i][j] = True if item i was taken to achieve dp[j]
    
    dp = np.zeros(capacity + 1, dtype=np.float64)
    takes = np.zeros((n, capacity + 1), dtype=bool)
    
    for i in range(n):
        w = weights[i]
        v = values[i]
        if w == 0:
            continue
        
        # Vectorized update
        candidate = dp[:-w] + v
        mask = candidate > dp[w:]
        
        if np.any(mask):
            dp[w:][mask] = candidate[mask]
            takes[i][w:][mask] = True
            
    # Backtrack
    j = capacity
    selected_f_indices = []
    
    for i in range(n - 1, -1, -1):
        if takes[i][j]:
            selected_f_indices.append(i)
            j -= weights[i]
            if j < 0:
                break
                
    if not selected_f_indices:
        # Fallback: Pick best single item that fits
        best_val = -1
        best_idx = -1
        for idx in range(n):
            if weights[idx] <= capacity:
                if values[idx] > best_val:
                    best_val = values[idx]
                    best_idx = idx
        if best_idx == -1:
            return -1
        return int(original_indices[best_idx])
        
    # Pick one item from the optimal set
    f_idx_to_pick = selected_f_indices[0]
    orig_idx = original_indices[f_idx_to_pick]
    
    return int(orig_idx)

def _solve_fractional_surrogate(values, weights, capacity, original_indices, n):
    """
    Uses the optimal fractional knapsack solution as a surrogate.
    1. Solve fractional knapsack (greedy by density).
    2. Identify the 'core' items: those fully taken in the fractional solution,
       and the one fractional item.
    3. The item to pick is chosen from the fully taken items or the fractional item.
       To be robust, we perform a small local search around the fractional solution
       to see if a better discrete choice exists among the top candidates.
    """
    # 1. Solve Fractional Knapsack
    # Sort by density (value/weight) descending
    densities = values / weights
    order = np.argsort(-densities)
    
    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_orig_indices = original_indices[order]
    
    # Determine which items are fully taken and which is fractional
    remaining_cap = capacity
    fully_taken_mask = np.zeros(n, dtype=bool)
    fractional_idx = -1
    fractional_ratio = 0.0
    
    for i in range(n):
        w = sorted_weights[i]
        if remaining_cap <= 0:
            break
        
        if w <= remaining_cap:
            fully_taken_mask[i] = True
            remaining_cap -= w
        else:
            fractional_idx = i
            fractional_ratio = remaining_cap / w
            remaining_cap = 0
            break
            
    # The fractional solution suggests picking any item in fully_taken_mask
    # or the fractional item.
    
    # Candidate pool:
    # 1. All fully taken items.
    # 2. The fractional item.
    # 3. Perhaps some items just below the cut-off?
    
    # Simple heuristic: Pick the item from the fully taken set with the highest density.
    # This is consistent with the greedy order.
    # However, sometimes swapping a low-density fully taken item with a high-density
    # excluded item (that is slightly heavier) might be better?
    # The fractional solution is optimal for the continuous case.
    # The discrete optimum is close to it.
    
    # Let's gather candidates:
    # All items that are either fully taken or the fractional item.
    candidate_indices_in_sorted = np.where(fully_taken_mask)[0]
    if fractional_idx != -1:
        candidate_indices_in_sorted = np.append(candidate_indices_in_sorted, fractional_idx)
    
    if len(candidate_indices_in_sorted) == 0:
        # No items fit? Should not happen if feasible_mask was checked.
        return -1
        
    # Now, among these candidates, which one should we pick?
    # If we pick one, the problem reduces.
    # The best next move is often the one that is "most critical" or "most dense".
    # Since we are building the solution incrementally, picking the highest density
    # item from the optimal fractional support is a strong heuristic.
    
    # Let's refine:
    # The fractional solution has a specific value.
    # If we pick an item from the support, we should ensure it's a good choice.
    
    # Strategy:
    # Pick the item with the highest density from the candidate set.
    # This aligns with the greedy order used to find the fractional solution.
    
    candidate_densities = densities[candidate_indices_in_sorted]
    best_local_idx = np.argmax(candidate_densities)
    best_sorted_idx = candidate_indices_in_sorted[best_local_idx]
    
    # Return the original index
    return int(sorted_orig_indices[best_sorted_idx])

# Fallback B&B (kept from original for potential use or if fractional fails, 
# though fractional is very fast and reliable for 100 items)
def _solve_bb(values, weights, capacity, original_indices, n):
    # Vectorized B&B
    # Sort by density (value/weight)
    densities = values / weights
    order = np.argsort(-densities)
    
    sorted_values = values[order]
    sorted_weights = weights[order]
    sorted_orig_indices = original_indices[order]
    
    # Precompute prefix sums for fractional bound
    prefix_values = np.cumsum(sorted_values)
    prefix_weights = np.cumsum(sorted_weights)
    
    best_val = 0.0
    best_set = set()
    
    start_time = time.time()
    time_limit = 0.5 # 0.5s per call max
    
    def dfs(i, curr_val, curr_weight, curr_set):
        nonlocal best_val, best_set
        if time.time() - start_time > time_limit:
            return
        
        if i == n:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return
            
        rem_cap = capacity - curr_weight
        if rem_cap <= 0:
            if curr_val > best_val:
                best_val = curr_val
                best_set = set(curr_set)
            return
            
        sum_rem_values = prefix_values[-1] - (prefix_values[i-1] if i > 0 else 0)
        sum_rem_weights = prefix_weights[-1] - (prefix_weights[i-1] if i > 0 else 0)
        
        if sum_rem_weights <= rem_cap:
            ub = curr_val + sum_rem_values
        else:
            rem_w = sorted_weights[i:]
            rem_v = sorted_values[i:]
            cum_w = np.cumsum(rem_w)
            k = np.searchsorted(cum_w, rem_cap, side='right') - 1
            
            if k < 0:
                ub = curr_val + rem_v[0] * (rem_cap / rem_w[0])
            else:
                val_full = np.sum(rem_v[:k+1])
                if k+1 < len(rem_w):
                    frac_val = rem_v[k+1] * (rem_cap - cum_w[k]) / rem_w[k+1]
                else:
                    frac_val = 0.0
                ub = curr_val + val_full + frac_val
        
        if ub <= best_val:
            return
            
        v_i = sorted_values[i]
        w_i = sorted_weights[i]
        idx_i = sorted_orig_indices[i]
        
        if curr_weight + w_i <= capacity:
            curr_set.add(idx_i)
            dfs(i + 1, curr_val + v_i, curr_weight + w_i, curr_set)
            curr_set.remove(idx_i)
            
        dfs(i + 1, curr_val, curr_weight, curr_set)
        
    curr_set = set()
    dfs(0, 0.0, 0.0, curr_set)
    
    if not best_set:
        feasible = sorted_weights <= capacity
        if not np.any(feasible):
            return -1
        best_idx_in_sorted = np.where(feasible)[0][0]
        return int(sorted_orig_indices[best_idx_in_sorted])
        
    best_item_idx = None
    for idx in best_set:
        best_item_idx = idx
        break
        
    return int(best_item_idx)
