"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
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
            ap_list = list(range(start, n + 1, step))
            if ap_list:
                aps.append(ap_list)
    
    m = len(aps)
    mat = np.zeros((m, n), dtype=np.int8)
    col_to_rows = [[] for _ in range(n)]
    
    for i, ap in enumerate(aps):
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
    signs = np.array([1 if x == 0 else -1 for x in tm[:n]], dtype=np.float64)
    return signs

def _compute_cost(sums):
    """Compute max absolute sum from a vector of AP sums."""
    return np.max(np.abs(sums))

def _sa_run(ap_mat, col_to_rows, n, rng, time_limit, T_start, T_end, alpha, initial_signs=None):
    """
    Run a single instance of Simulated Annealing with incremental updates.
    Uses a smarter cost tracking: track the top few max values to avoid full
    recomputation when possible.
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
    
    # Track the current max and second max for faster cost updates
    # We'll recompute cost periodically and use incremental updates in between
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        old_sign = current_signs[idx]
        new_sign = -old_sign
        current_signs[idx] = new_sign
        
        # Update sums incrementally
        delta_sign = new_sign - old_sign  # +/-2
        affected_rows = col_to_rows[idx]
        
        if affected_rows:
            current_sums[affected_rows] += delta_sign
        
        # Calculate new cost - for speed, we can check if the flipped value
        # could have affected the max
        new_cost = _compute_cost(current_sums)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
        else:
            delta = new_cost - current_cost
            # Accept with probability exp(-delta / T)
            if delta > 0:
                if rng.random() < np.exp(-delta / T):
                    pass  # Accept
                else:
                    # Reject: revert
                    current_signs[idx] = old_sign
                    if affected_rows:
                        current_sums[affected_rows] -= delta_sign
                    current_cost = _compute_cost(current_sums)
            else:
                pass

        T *= alpha
    
    return best_signs, best_cost

def _local_search(ap_mat, col_to_rows, n, signs, max_iters=5000):
    """
    Greedy local search: try flipping each position and accept if it reduces max |sum|.
    Uses incremental updates for speed.
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

def _multi_flip_sa(ap_mat, col_to_rows, n, rng, time_limit, T_start, T_end, alpha, initial_signs=None):
    """
    SA variant that flips 2-4 positions at a time to escape local minima.
    """
    if initial_signs is not None:
        current_signs = initial_signs.copy()
    else:
        current_signs = rng.choice([-1, 1], size=n).astype(np.float64)
    
    current_sums = ap_mat @ current_signs
    current_cost = _compute_cost(current_sums)
    
    best_signs = current_signs.copy()
    best_cost = current_cost
    
    T = T_start
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit and T > T_end:
        # Decide how many positions to flip (1-4)
        num_flips = rng.integers(1, 5)
        indices = rng.choice(n, size=num_flips, replace=False)
        
        old_signs = current_signs[indices].copy()
        new_signs = -old_signs
        current_signs[indices] = new_signs
        
        # Update sums
        for idx in indices:
            delta_sign = current_signs[idx] - old_signs[np.where(indices == idx)[0][0]]
            affected_rows = col_to_rows[idx]
            if affected_rows:
                current_sums[affected_rows] += delta_sign
        
        new_cost = _compute_cost(current_sums)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = current_signs.copy()
        else:
            delta = new_cost - current_cost
            if delta > 0:
                if rng.random() < np.exp(-delta / T):
                    pass  # Accept
                else:
                    # Reject: revert
                    current_signs[indices] = old_signs
                    for idx in indices:
                        delta_sign = old_signs[np.where(indices == idx)[0][0]] - current_signs[idx]
                        affected_rows = col_to_rows[idx]
                        if affected_rows:
                            current_sums[affected_rows] += delta_sign
                    current_cost = _compute_cost(current_sums)
            else:
                pass

        T *= alpha
    
    return best_signs, best_cost

def construct():
    n = 256
    ap_mat, col_to_rows = _get_cached_data(n)
    
    total_start = time.time()
    max_total_time = 55.0  # Leave 5s buffer for overhead
    
    rng = np.random.default_rng(42)
    
    best_global_signs = None
    best_global_cost = np.inf
    
    # Generate diverse initializations
    initial_strategies = []
    
    # Strategy 1: Thue-Morse
    initial_strategies.append(_thue_morse(n))
    
    # Strategy 2: Random
    initial_strategies.append(rng.choice([-1, 1], size=n).astype(np.float64))
    
    # Strategy 3: Alternating
    alt = np.ones(n, dtype=np.float64)
    alt[1::2] = -1
    initial_strategies.append(alt)
    
    # Strategy 4: Random with different seed
    initial_strategies.append(rng.choice([-1, 1], size=n).astype(np.float64))
    
    # Strategy 5: Block pattern
    block = np.ones(n, dtype=np.float64)
    for i in range(0, n, 8):
        block[i:i+4] = -1
    initial_strategies.append(block)
    
    # Strategy 6: Random walk
    walk = np.zeros(n, dtype=np.float64)
    walk[0] = 1
    for i in range(1, n):
        walk[i] = walk[i-1] * rng.choice([-1, 1])
    initial_strategies.append(walk)
    
    num_restarts = len(initial_strategies)
    
    # Allocate time: 60% to SA, 30% to multi-flip SA, 10% to local search
    sa_time = max_total_time * 0.6
    multi_sa_time = max_total_time * 0.3
    ls_time = max_total_time * 0.1
    
    # Run single-flip SA for each strategy
    time_per_sa = sa_time / num_restarts
    
    for r in range(num_restarts):
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining <= 1.0:
            break
        
        init_signs = initial_strategies[r]
        
        T_start = 5.0
        T_end = 0.05
        alpha = 0.9997
        
        best_signs, best_cost = _sa_run(
            ap_mat, col_to_rows, n, rng, 
            time_per_sa,
            T_start, T_end, alpha,
            initial_signs=init_signs
        )
        
        if best_cost < best_global_cost:
            best_global_cost = best_cost
            best_global_signs = best_signs.copy()
    
    # Run multi-flip SA on the best solution found so far
    if best_global_signs is not None and time.time() - total_start < max_total_time - 1.0:
        elapsed = time.time() - total_start
        remaining = max_total_time - elapsed
        if remaining > 1.0:
            multi_best_signs, multi_best_cost = _multi_flip_sa(
                ap_mat, col_to_rows, n, rng,
                min(multi_sa_time, remaining - 1.0),
                8.0, 0.1, 0.9998,
                initial_signs=best_global_signs
            )
            if multi_best_cost < best_global_cost:
                best_global_cost = multi_best_cost
                best_global_signs = multi_best_signs.copy()
    
    # Final local search pass on the best solution
    remaining = max_total_time - (time.time() - total_start)
    if remaining > 1.0 and best_global_signs is not None:
        ls_signs, ls_cost = _local_search(ap_mat, col_to_rows, n, best_global_signs, max_iters=10000)
        if ls_cost < best_global_cost:
            best_global_signs = ls_signs
            best_global_cost = ls_cost
    
    if best_global_signs is None:
        best_global_signs = np.ones(n)
        
    return [int(x) for x in best_global_signs]

# EVOLVE-BLOCK-END
