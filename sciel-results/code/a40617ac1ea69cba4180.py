"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, P):
    """Calculate makespan for a given permutation using vectorized operations."""
    n = len(perm)
    if n == 0:
        return 0.0
    
    m = P.shape[1]
    # P_perm is the matrix of processing times in the order of perm
    P_perm = P[perm]
    
    # C[i, j] is the completion time of the i-th job in perm on the j-th machine
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
    # We can compute this iteratively.
    # To vectorize partially, we can use a loop over jobs but vectorize over machines?
    # Or just use a simple loop which is often fast enough for N < 1000.
    # Let's try a slightly optimized loop.
    
    C_prev = np.zeros(m)
    C_curr = np.zeros(m)
    
    for i in range(n):
        # C_curr[0] = C_prev[0] + P_perm[i, 0]
        # C_curr[j] = max(C_prev[j], C_curr[j-1]) + P_perm[i, j]
        
        # We can't fully vectorize the dependency C_curr[j-1] -> C_curr[j]
        # But we can do it in a tight loop.
        # Alternative: use numpy accumulate? No, the max dependency is complex.
        
        # Let's stick to the loop but ensure we are using numpy scalars/arrays efficiently
        # Actually, for small N (e.g. < 100), Python loop is fine.
        # For larger N, we might need C extension or better vectorization.
        # Given the constraints, let's optimize the Python loop.
        
        c_prev_0 = C_prev[0]
        c_curr_0 = c_prev_0 + P_perm[i, 0]
        C_curr[0] = c_curr_0
        
        for j in range(1, m):
            # C_curr[j] = max(C_prev[j], C_curr[j-1]) + P_perm[i, j]
            val = C_prev[j]
            if C_curr[j-1] > val:
                val = C_curr[j-1]
            C_curr[j] = val + P_perm[i, j]
            
        C_prev, C_curr = C_curr, C_prev
        
    return C_prev[-1]

def _neh_initial(P):
    """Construct initial solution using NEH heuristic."""
    n, m = P.shape
    totals = P.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = np.inf
        # Optimization: limit insertion positions to reduce cost? 
        # Standard NEH checks all. 
        # For speed, we can check all if n is small.
        for i in range(len(perm) + 1):
            new_perm = perm[:i] + [job] + perm[i:]
            ms = _makespan(new_perm, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = i
        perm.insert(best_pos, job)
        
    return np.array(perm)

def _palmer_initial(P):
    """Construct initial solution using Palmer's heuristic."""
    n, m = P.shape
    # W_j = (m+1) * p_j1 - sum_{k=1}^m (m+1-k) * p_jk
    # Actually Palmer index: W_j = sum_{k=1}^m (m+1-2k) * p_jk ?
    # Common definition: W_j = (m+1) * sum_{k=1}^m p_jk - 2 * sum_{k=1}^m k * p_jk
    # Let's use the standard Palmer index: W_j = sum_{k=1}^m (m+1-2k) * p_jk
    # Wait, different sources define it slightly differently.
    # Let's use: W_j = sum_{k=1}^m (m+1-k) * p_jk - sum_{k=1}^m k * p_jk
    # This is equivalent to sum (m+1-2k) p_jk
    
    weights = np.arange(1, m+1)
    # W_j = sum_{k} (m+1-k) P[j,k] - sum_{k} k P[j,k]
    # W_j = sum_{k} (m+1-2k) P[j,k]
    coeffs = (m + 1 - 2 * weights)
    W = P @ coeffs
    
    # Sort by descending W
    order = np.argsort(-W)
    return order

def _lpt_initial(P):
    """Longest Processing Time (total) heuristic."""
    totals = P.sum(axis=1)
    return np.argsort(-totals)

def _spt_initial(P):
    """Shortest Processing Time (total) heuristic."""
    totals = P.sum(axis=1)
    return np.argsort(totals)

def _random_initial(P):
    n = P.shape[0]
    perm = np.arange(n)
    np.random.shuffle(perm)
    return perm

def _insertion_local_search(perm, P, time_limit, start_time):
    """
    Improve permutation using insertion moves (remove job i, insert at j).
    This covers 2-opt and Or-opt(1) effectively.
    """
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, P)
    
    # We will iterate until no improvement or time limit
    # To speed up, we can pre-compute some deltas? 
    # For now, simple re-evaluation.
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        current_perm = best_perm.copy()
        current_ms = best_ms
        
        # Try all insertion moves
        # Remove job at i, insert at j
        # i ranges 0..n-1
        # j ranges 0..n-1 (in the list with i removed, length n-1)
        
        for i in range(n):
            job = current_perm[i]
            # Create remaining list
            # To avoid creating many lists, we can simulate?
            # For n < 1000, list creation is okay.
            remaining = list(current_perm[:i] + current_perm[i+1:])
            
            for j in range(len(remaining) + 1):
                # Skip if j == i (insert back at same place) 
                # Note: if j < i, index is same. If j >= i, index is j in new list?
                # If we insert at j in 'remaining', the new position in full list is:
                # if j < i: pos j
                # if j >= i: pos j+1
                # Original pos was i.
                # So skip if (j < i and j == i) -> impossible
                # Skip if (j >= i and j+1 == i) -> j = i-1
                # So skip j == i-1? 
                # Let's just check if new_perm == current_perm
                
                new_perm = remaining[:j] + [job] + remaining[j:]
                
                if new_perm == list(current_perm):
                    continue
                
                ms = _makespan(new_perm, P)
                if ms < current_ms - 1e-9:
                    current_perm = np.array(new_perm)
                    current_ms = ms
                    improved = True
                    # First improvement strategy: break and restart pass
                    break
            
            if improved:
                break
        
        if improved:
            best_perm = current_perm
            best_ms = current_ms
        else:
            # No improvement, stop
            break
            
    return best_perm

def _simulated_annealing(perm, P, time_limit, start_time):
    """
    Simulated annealing to escape local optima.
    """
    n = len(perm)
    if n < 2:
        return perm
    
    current_perm = perm.copy()
    current_ms = _makespan(current_perm, P)
    
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    # Initial temperature
    # Estimate max delta
    max_p = P.max()
    n_machines = P.shape[1]
    # Rough upper bound on makespan change
    T = max_p * n_machines * 0.1 
    if T == 0:
        T = 1.0
    
    T_min = 0.01
    cooling_rate = 0.99
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
            
        # Generate neighbor: random swap or insertion
        if random.random() < 0.5:
            # Swap
            i, j = random.sample(range(n), 2)
            new_perm = current_perm.copy()
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        else:
            # Insertion (remove i, insert at j)
            i = random.randint(0, n-1)
            job = current_perm[i]
            remaining = list(current_perm[:i] + current_perm[i+1:])
            j = random.randint(0, len(remaining))
            new_perm = np.array(remaining[:j] + [job] + remaining[j:])
        
        new_ms = _makespan(new_perm, P)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = current_perm.copy()
                best_ms = current_ms
        else:
            # Accept with probability
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_perm = new_perm
                current_ms = new_ms
        
        T *= cooling_rate
        
    return best_perm

def order_jobs(processing_times):
    P = processing_times
    n, m = P.shape
    
    start_time = time.time()
    # Safety margin: 10s
    time_limit = 140.0 
    
    # 1. Generate multiple initial solutions
    heuristics = [_neh_initial, _palmer_initial, _lpt_initial, _spt_initial, _random_initial]
    
    best_initial_perm = None
    best_initial_ms = np.inf
    
    for h in heuristics:
        if time.time() - start_time > time_limit - 10:
            break
        perm = h(P)
        ms = _makespan(perm, P)
        if ms < best_initial_ms:
            best_initial_ms = ms
            best_initial_perm = perm
    
    if best_initial_perm is None:
        best_initial_perm = _neh_initial(P)
        best_initial_ms = _makespan(best_initial_perm, P)
        
    current_best_perm = best_initial_perm
    current_best_ms = best_initial_ms
    
    # 2. Local Search on the best initial
    if time.time() - start_time < time_limit - 5:
        ls_perm = _insertion_local_search(best_initial_perm, P, time_limit, start_time)
        ls_ms = _makespan(ls_perm, P)
        if ls_ms < current_best_ms:
            current_best_perm = ls_perm
            current_best_ms = ls_ms
            
    # 3. Simulated Annealing to explore further
    # Run SA for remaining time
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        sa_perm = _simulated_annealing(current_best_perm, P, time_limit, start_time)
        sa_ms = _makespan(sa_perm, P)
        if sa_ms < current_best_ms:
            current_best_perm = sa_perm
            current_best_ms = sa_ms
            
    # 4. Final Local Search polish
    if time.time() - start_time < time_limit - 1:
        final_perm = _insertion_local_search(current_best_perm, P, time_limit, start_time)
        final_ms = _makespan(final_perm, P)
        if final_ms < current_best_ms:
            current_best_perm = final_perm
            current_best_ms = final_ms
            
    return current_best_perm

# EVOLVE-BLOCK-END
