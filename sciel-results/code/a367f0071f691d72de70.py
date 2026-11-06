"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

_AP_MAT_CACHE = None
_COL_TO_ROWS_CACHE = None

def _build_ap_matrix(n=256):
    """
    Build a dense binary incidence matrix where each row corresponds to an
    arithmetic progression within [1, n] (1-indexed) and each column to a position.
    Entry (i, j) is 1 if position j+1 is in AP i.
    Also returns a list where col_to_rows[j] contains indices of APs containing column j.
    """
    aps = []
    for start in range(1, n + 1):
        for step in range(1, n - start + 1):
            # Generate the AP: start, start+step, ..., <= n
            ap_list = list(range(start, n + 1, step))
            if ap_list:
                aps.append(ap_list)
    
    m = len(aps)
    mat = np.zeros((m, n), dtype=np.int8)
    col_to_rows = [[] for _ in range(n)]
    
    for i, ap in enumerate(aps):
        # Vectorized assignment
        indices = np.array(ap, dtype=np.int32) - 1
        mat[i, indices] = 1
        for idx in indices:
            col_to_rows[idx].append(i)
            
    return mat, col_to_rows

def _get_cached_data(n=256):
    global _AP_MAT_CACHE, _COL_TO_ROWS_CACHE
    if _AP_MAT_CACHE is None:
        _AP_MAT_CACHE, _COL_TO_ROWS_CACHE = _build_ap_matrix(n)
    return _AP_MAT_CACHE, _COL_TO_ROWS_CACHE

def _sa_run(ap_mat, col_to_rows, n, rng, time_limit, T_start, T_end, alpha):
    """
    Run a single instance of Simulated Annealing with incremental updates.
    """
    # Initial solution: random signs
    current_signs = rng.choice([-1, 1], size=n).astype(np.float64)
    
    # Initial sums
    current_sums = ap_mat @ current_signs
    current_cost = np.max(np.abs(current_sums))
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    T = T_start
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        old_sign = current_signs[idx]
        new_sign = -old_sign
        current_signs[idx] = new_sign
        
        # Update sums incrementally
        # The change in sum for AP i is (new_sign - old_sign) * ap_mat[i, idx]
        # Since signs are +/-1, new_sign - old_sign is +/-2
        delta_sign = new_sign - old_sign # +/-2
        affected_rows = col_to_rows[idx]
        
        if affected_rows:
            # Update the specific rows
            current_sums[affected_rows] += delta_sign
        
        # Calculate new cost
        # np.max on 32k floats is fast
        new_cost = np.max(np.abs(current_sums))
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
        else:
            delta = new_cost - current_cost
            # Accept with probability exp(-delta / T)
            if delta > 0:
                # Optimization: if delta is large, exp(-delta/T) is tiny
                if rng.random() < np.exp(-delta / T):
                    pass # Accept
                else:
                    # Reject: revert
                    current_signs[idx] = old_sign
                    current_sums[affected_rows] -= delta_sign
            else:
                # delta <= 0, always accept (already handled in if, but logic covers it)
                pass

        T *= alpha
        
    return best_signs, best_cost

def construct():
    n = 256
    ap_mat, col_to_rows = _get_cached_data(n)
    
    total_start = time.time()
    max_total_time = 55.0 # Leave 5s buffer for overhead
    
    rng = np.random.default_rng(42)
    
    best_global_signs = None
    best_global_cost = np.inf
    
    # Strategy: Multi-restart SA
    # With incremental updates, we can run many more iterations.
    # Estimate: Update ~100-200 rows + np.max ~ 10-20us.
    # Total ~ 50us per iteration? 20,000 iterations/sec.
    # 50s -> 1,000,000 iterations.
    
    num_restarts = 10
    time_per_restart = (max_total_time - (time.time() - total_start)) / num_restarts
    
    for r in range(num_restarts):
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining <= 1.0:
            break
            
        # Tune parameters
        # Higher T_start for exploration
        T_start = 10.0
        T_end = 0.1
        # Alpha: we want to cool down over the time limit
        # If we assume ~20000 iters/sec and time_per_restart is ~5s, N ~ 100,000
        # alpha = (T_end/T_start)^(1/N)
        # Let's just use a conservative alpha that cools down enough
        alpha = 0.9995 
        
        best_signs, best_cost = _sa_run(ap_mat, col_to_rows, n, rng, time_per_restart, T_start, T_end, alpha)
        
        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_signs = best_signs.copy()
            
        if time.time() - total_start > max_total_time - 1.0:
            break
            
    if best_global_signs is None:
        best_global_signs = np.ones(n)
        
    return [int(x) for x in best_global_signs]

# EVOLVE-BLOCK-END
