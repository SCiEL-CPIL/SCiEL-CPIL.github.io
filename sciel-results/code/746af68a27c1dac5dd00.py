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

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (0s and 1s), then convert to -1/+1."""
    tm = [0]
    for i in range(n - 1):
        tm.append(tm[-1] ^ (i & 1))
    # Convert 0->1, 1->-1
    signs = np.array([1 if x == 0 else -1 for x in tm[:n]], dtype=np.float64)
    return signs

def _compute_cost(sums):
    """Compute max absolute sum."""
    return int(np.max(np.abs(sums)))

def _sa_run(ap_mat, col_to_rows, n, rng, time_limit, T_start, T_end, alpha, initial_signs=None):
    """
    Run a single instance of Simulated Annealing with incremental updates.
    """
    if initial_signs is not None:
        current_signs = initial_signs.copy()
    else:
        current_signs = rng.choice([-1, 1], size=n).astype(np.float64)
    
    # Initial sums
    current_sums = ap_mat @ current_signs
    current_cost = _compute_cost(current_sums)
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    T = T_start
    start_time = time.time()
    
    # Track recent improvements for adaptive cooling
    last_improve_time = start_time
    no_improve_count = 0
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Pick a random index to flip
        idx = int(rng.integers(0, n))
        
        old_sign = current_signs[idx]
        new_sign = -old_sign
        current_signs[idx] = new_sign
        
        # Update sums incrementally
        delta_sign = new_sign - old_sign  # +/-2
        affected_rows = col_to_rows[idx]
        
        if affected_rows:
            current_sums[affected_rows] += delta_sign
        
        # Calculate new cost
        new_cost = _compute_cost(current_sums)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve_count = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
                last_improve_time = time.time()
        else:
            delta = new_cost - current_cost
            no_improve_count += 1
            # Accept with probability exp(-delta / T)
            if delta > 0:
                if rng.random() < np.exp(-delta / T):
                    pass  # Accept
                else:
                    # Reject: revert
                    current_signs[idx] = old_sign
                    current_sums[affected_rows] -= delta_sign
            else:
                pass

        # Adaptive cooling: if no improvement for a while, cool faster
        elapsed = time.time() - start_time
        if no_improve_count > 1000 or (elapsed - last_improve_time > 2.0):
            T *= 0.98
            no_improve_count = 0
            last_improve_time = time.time()
        else:
            T *= alpha
        
    return best_signs, best_cost

def _local_search(ap_mat, col_to_rows, n, signs, max_iters=5000):
    """
    Greedy local search: try flipping each position and accept if it reduces max |sum|.
    """
    current_signs = signs.copy()
    current_sums = ap_mat @ current_signs
    current_cost = _compute_cost(current_sums)
    
    improved = True
    iterations = 0
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        for idx in range(n):
            old_sign = current_signs[idx]
            new_sign = -old_sign
            current_signs[idx] = new_sign
            
            delta_sign = new_sign - old_sign
            affected_rows = col_to_rows[idx]
            if affected_rows:
                current_sums[affected_rows] += delta_sign
            
            new_cost = _compute_cost(current_sums)
            
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
            else:
                # Revert
                current_signs[idx] = old_sign
                if affected_rows:
                    current_sums[affected_rows] -= delta_sign
    
    return current_signs, current_cost

def construct():
    n = 256
    ap_mat, col_to_rows = _get_cached_data(n)
    
    total_start = time.time()
    max_total_time = 55.0  # Leave 5s buffer for overhead
    
    rng = np.random.default_rng(42)
    
    best_global_signs = None
    best_global_cost = np.inf
    
    # Multiple strategies:
    # 1. Thue-Morse initial
    # 2. Random initial
    # 3. Alternating initial
    # 4. Another random with different seed
    
    initial_strategies = []
    
    # Strategy 1: Thue-Morse
    initial_strategies.append(_thue_morse(n))
    
    # Strategy 2: Random
    initial_strategies.append(rng.choice([-1, 1], size=n).astype(np.float64))
    
    # Strategy 3: Alternating
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1
    initial_strategies.append(alt)
    
    # Strategy 4: Another random
    initial_strategies.append(rng.choice([-1, 1], size=n).astype(np.float64))
    
    # Run SA for each strategy
    num_restarts = len(initial_strategies)
    
    for r in range(num_restarts):
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining <= 1.0:
            break
        
        init_signs = initial_strategies[r]
        
        # Tune parameters - use more aggressive cooling
        T_start = 3.0
        T_end = 0.01
        alpha = 0.9995
        
        # Use 60% of remaining time for SA, 40% for local search
        sa_time = min(remaining * 0.6, 15.0)
        
        best_signs, best_cost = _sa_run(
            ap_mat, col_to_rows, n, rng, 
            sa_time,
            T_start, T_end, alpha,
            initial_signs=init_signs
        )
        
        # Local search on the best found
        ls_time_budget = max(remaining - (time.time() - total_start) - 2.0, 1.0)
        ls_iters = min(int(ls_time_budget * 2000), 8000)
        ls_signs, ls_cost = _local_search(ap_mat, col_to_rows, n, best_signs, max_iters=ls_iters)
        
        if ls_cost < best_cost:
            best_signs, best_cost = ls_signs, ls_cost
        
        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_signs = best_signs.copy()
            
        if time.time() - total_start > max_total_time - 1.0:
            break
    
    # If time remains, do more local search on the best solution
    remaining = max_total_time - (time.time() - total_start)
    if remaining > 2.0 and best_global_signs is not None:
        ls_iters = min(int(remaining * 1500), 15000)
        ls_signs, ls_cost = _local_search(ap_mat, col_to_rows, n, best_global_signs, max_iters=ls_iters)
        if ls_cost < best_global_cost:
            best_global_signs = ls_signs
            best_global_cost = ls_cost
    
    if best_global_signs is None:
        best_global_signs = np.ones(n)
        
    return [int(x) for x in best_global_signs]

# EVOLVE-BLOCK-END
