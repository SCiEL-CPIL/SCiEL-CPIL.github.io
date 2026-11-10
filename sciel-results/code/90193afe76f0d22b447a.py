import numpy as np
import time
import sys

sys.setrecursionlimit(10000)

# Global deadline guard
GLOBAL_START_TIME = time.time()
GLOBAL_DEADLINE = GLOBAL_START_TIME + 140.0  # 10s safety margin before 150s kill

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check global deadline
    if time.time() > GLOBAL_DEADLINE:
        # Fallback to a very fast greedy choice to ensure return
        weights = np.array(remaining_weights, dtype=float)
        values = np.array(remaining_values, dtype=float)
        feasible = (weights <= remaining_capacity) & (weights > 0)
        if not np.any(feasible):
            # Try zero weight positive value
            zw = (weights == 0) & (values > 0)
            if np.any(zw):
                return int(np.where(zw)[0][0])
            return -1
        f_idx = np.where(feasible)[0]
        f_v = values[feasible]
        # Pick highest value
        best_local = np.argmax(f_v)
        return int(f_idx[best_local])

    weights = np.array(remaining_weights, dtype=float)
    values = np.array(remaining_values, dtype=float)
    
    # Handle zero weight, positive value items immediately
    zero_w_pos_v_mask = (weights == 0) & (values > 0)
    if np.any(zero_w_pos_v_mask):
        return int(np.where(zero_w_pos_v_mask)[0][0])
    
    # Handle negative weight, positive value items (always beneficial)
    neg_w_pos_v_mask = (weights < 0) & (values > 0)
    if np.any(neg_w_pos_v_mask):
        return int(np.where(neg_w_pos_v_mask)[0][0])
    
    # Filter feasible items
    feasible_mask = (weights > 0) & (weights <= remaining_capacity)
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_mask]
    f_values = values[feasible_mask]
    f_n = len(f_weights)
    
    if f_n == 0:
        return -1
        
    # --- Hybrid Strategy: DP for small/medium, B&B for large ---
    
    # Determine if DP is feasible
    # DP complexity is O(N * C_scaled).
    # We want to keep operations under ~20M for speed.
    # C is remaining_capacity.
    # Scale weights to integers.
    
    # Simple heuristic for scaling:
    # If weights are already integers or close, use them.
    # Otherwise, scale up.
    # Let's try to determine a scale factor such that C_scaled is reasonable.
    
    # If f_n is small (< 30), B&B is usually very fast.
    # If f_n is large, DP is better if C is small enough.
    
    # Let's estimate DP cost.
    # Assume we scale weights so that max_weight_scaled is around 100-1000?
    # Or just use integer weights if possible.
    
    # Let's try a fixed scale based on precision.
    # If weights are floats, we might lose precision.
    # Let's check if weights are effectively integers.
    
    # Strategy:
    # 1. If f_n <= 25, use B&B (exact, fast enough).
    # 2. Else, try DP with scaling.
    
    if f_n <= 25:
        return _solve_bb(f_weights, f_values, remaining_capacity, feasible_indices)
    
    # Try DP
    # Scale weights to integers
    # We want to preserve relative differences.
    # Let's find a scale factor.
    # If weights are like 0.1, 0.2, scale by 10.
    # If weights are like 1.0, 1.1, scale by 10.
    # Let's try scale = 100.
    # C_scaled = remaining_capacity * 100.
    # If C is 12.5, C_scaled = 1250.
    # N=100, Ops = 125,000. Very fast.
    
    scale = 100
    w_int = (f_weights * scale).astype(np.int64)
    C_int = int(remaining_capacity * scale)
    
    # Check if scaling is valid (no overflow, etc.)
    if C_int > 50000 or f_n * C_int > 50_000_000:
        # DP too large, fall back to B&B or Greedy
        # B&B might be slow if N is large, but with good bounds maybe ok?
        # Or just use Greedy?
        # Let's try B&B with a time limit.
        return _solve_bb(f_weights, f_values, remaining_capacity, feasible_indices)
    
    # Perform DP
    # dp[j] = max value using capacity j
    dp = np.zeros(C_int + 1, dtype=np.float64)
    
    # To backtrack, we need to store which items were picked.
    # Storing a full table of booleans takes N * C_int memory.
    # 100 * 12500 = 1.25M bools. OK.
    takes = np.zeros((f_n, C_int + 1), dtype=bool)
    
    # Optimization: Check if any weight is 0 after scaling?
    # We filtered w > 0, so w_int >= 1 (assuming scale is big enough).
    # If w_int is 0, it means weight was very small.
    # If w_int == 0, we should pick it if value > 0.
    # But we filtered w > 0.
    # If w * scale < 1, it becomes 0.
    # Let's handle this: if any w_int == 0, pick the one with highest value?
    # Actually, if w_int == 0, it fits in any capacity.
    zero_scaled = np.where(w_int == 0)[0]
    if len(zero_scaled) > 0:
        # Pick the one with highest value among zero-scaled
        best_z = zero_scaled[np.argmax(f_values[zero_scaled])]
        return int(feasible_indices[best_z])
    
    # DP Loop
    for i in range(f_n):
        w = w_int[i]
        v = f_values[i]
        if w > C_int:
            continue
        # Update dp from C_int down to w
        # dp[j] = max(dp[j], dp[j-w] + v)
        # Vectorized:
        # prev = dp[:C_int-w+1]
        # cand = prev + v
        # cur = dp[w:]
        # mask = cand > cur
        # dp[w:][mask] = cand[mask]
        # takes[i][w:][mask] = True
        
        # Note: dp[:C_int-w+1] has length C_int-w+1.
        # dp[w:] has length C_int-w+1.
        
        prev_vals = dp[:C_int - w + 1]
        cand_vals = prev_vals + v
        cur_slice = dp[w:]
        
        mask = cand_vals > cur_slice
        if np.any(mask):
            # Update dp
            # cur_slice is a view.
            # cur_slice[mask] = cand_vals[mask]
            # This modifies dp in place.
            cur_slice[mask] = cand_vals[mask]
            takes[i][w:][mask] = True
    
    # Backtrack
    # Start from full capacity C_int
    j = C_int
    selected_indices = []
    
    for i in range(f_n - 1, -1, -1):
        if takes[i][j]:
            selected_indices.append(i)
            j -= w_int[i]
            if j < 0:
                break
    
    if not selected_indices:
        # Fallback: pick best single item
        # This shouldn't happen if there are feasible items
        best_local = np.argmax(f_values)
        return int(feasible_indices[best_local])
    
    # Pick one item from the optimal set
    # The list `selected_indices` contains indices into `f_weights` (0..f_n-1)
    # Map to original indices
    # We can pick any one.
    # Let's pick the first one found (which corresponds to the last item in the optimal set? No, backtracking goes N-1 to 0).
    # selected_indices[0] is the index of the item that was checked last (i=0..N-1 loop, so i=N-1 first).
    # It doesn't matter which one we pick, as long as it's in the optimal set.
    
    f_idx_to_pick = selected_indices[0]
    orig_idx = feasible_indices[f_idx_to_pick]
    
    return int(orig_idx)

def _solve_bb(f_weights, f_values, remaining_capacity, feasible_indices):
    n = len(f_weights)
    if n == 0:
        return -1
    
    # Sort by density descending
    densities = f_values / f_weights
    sorted_order = np.argsort(-densities)
    
    s_weights = f_weights[sorted_order]
    s_values = f_values[sorted_order]
    s_indices = feasible_indices[sorted_order]
    
    best_value = -1
    best_set = []
    
    start_time = time.time()
    time_limit = 0.5 # 500ms
    
    def fractional_bound(idx, curr_val, curr_weight):
        ub = curr_val
        rem_cap = remaining_capacity - curr_weight
        if rem_cap <= 0:
            return ub
            
        for j in range(idx, n):
            if s_weights[j] <= rem_cap:
                ub += s_values[j]
                rem_cap -= s_weights[j]
            else:
                if s_weights[j] > 0:
                    ub += s_values[j] * (rem_cap / s_weights[j])
                break
        return ub
    
    def dfs(idx, curr_val, curr_weight, curr_set):
        nonlocal best_value, best_set
        
        if time.time() - start_time > time_limit:
            return
            
        if idx == n:
            if curr_val > best_value:
                best_value = curr_val
                best_set = list(curr_set)
            return
            
        ub = fractional_bound(idx, curr_val, curr_weight)
        if ub <= best_value:
            return
            
        w = s_weights[idx]
        v = s_values[idx]
        
        # Include
        if curr_weight + w <= remaining_capacity:
            curr_set.append(s_indices[idx])
            dfs(idx + 1, curr_val + v, curr_weight + w, curr_set)
            curr_set.pop()
            
        # Exclude
        dfs(idx + 1, curr_val, curr_weight, curr_set)
        
    curr_set = []
    dfs(0, 0.0, 0.0, curr_set)
    
    if not best_set:
        # Fallback: greedy
        # Pick first feasible item in sorted order (highest density)
        return int(s_indices[0])
        
    return int(best_set[0])
