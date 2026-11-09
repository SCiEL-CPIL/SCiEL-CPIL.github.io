import time
import numpy as np

def _makespan(perm, pt):
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return 0.0
    
    # Use a 1D array for the previous machine's completion times
    # and update in-place if possible, but we need the previous values for the max.
    # So we need two arrays.
    
    prev_C = np.cumsum(pt[perm, 0])
    
    for k in range(1, m):
        curr_C = np.empty(n, dtype=np.float64)
        curr_C[0] = prev_C[0] + pt[perm[0], k]
        # Vectorized part is hard due to dependency on curr_C[i-1]
        for i in range(1, n):
            curr_C[i] = max(prev_C[i], curr_C[i-1]) + pt[perm[i], k]
        prev_C = curr_C
        
    return float(prev_C[-1])

def _neh_heuristic(pt, sort_key='sum'):
    n, m = pt.shape
    if sort_key == 'sum':
        order = np.argsort(-pt.sum(axis=1))
    elif sort_key == 'max':
        order = np.argsort(-pt.max(axis=1))
    elif sort_key == 'mean':
        order = np.argsort(-pt.mean(axis=1))
    else:
        order = np.argsort(-pt.sum(axis=1))
        
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        if not perm:
            perm = [job]
            continue
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return np.array(perm, dtype=np.int64)

def _local_search_insert(perm, pt, max_iters=100, deadline=None):
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() >= deadline:
            break
        
        # Try moving each job to a better position
        for i in range(n):
            if deadline is not None and time.time() >= deadline:
                break
            job = best_perm[i]
            # Remove job
            remaining = np.concatenate([best_perm[:i], best_perm[i+1:]])
            # Try inserting at all positions in remaining
            for j in range(len(remaining) + 1):
                new_perm = np.concatenate([remaining[:j], [job], remaining[j:]])
                new_ms = _makespan(new_perm, pt)
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = new_ms
                    improved = True
                    # Restart the outer loop? Or just continue?
                    # Continuing is faster, but might miss improvements.
                    # Let's break out of inner loop and restart outer to be safe?
                    # Actually, first improvement is usually enough.
                    break
            # If we found an improvement, we should probably restart the scan
            # to ensure we don't miss better moves for other jobs.
            if improved:
                break
                
    return best_perm

def _simulated_annealing(perm, pt, deadline, max_iters=2000):
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = perm.copy()
    current_ms = _makespan(current_perm, pt)
    best_perm = current_perm.copy()
    best_ms = current_ms
    
    start_temp = max(1.0, current_ms * 0.05)
    end_temp = 1e-4
    # Cooling rate to reach end_temp in max_iters
    if max_iters > 0:
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
    else:
        cooling_rate = 0.99
        
    temp = start_temp
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        if temp < end_temp:
            break
            
        # Generate neighbor: Random Insertion
        i = rng.integers(0, n)
        job = current_perm[i]
        remaining = np.concatenate([current_perm[:i], current_perm[i+1:]])
        j = rng.integers(0, len(remaining) + 1)
        
        if i == j or (i > 0 and i == j): # Handle index adjustment carefully
             # If we remove i, the positions after i shift left by 1.
             # If we insert at j in the remaining list, we need to map back to original indices?
             # No, we just construct the new perm.
             pass
             
        new_perm = np.concatenate([remaining[:j], [job], remaining[j:]])
        new_ms = _makespan(new_perm, pt)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms - 1e-9:
                best_perm = current_perm.copy()
                best_ms = current_ms
                
        temp *= cooling_rate
        
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    best_perm = None
    best_ms = float('inf')
    
    # 1. Generate multiple NEH variants
    variants = ['sum', 'max', 'mean']
    for v in variants:
        if time.time() >= deadline - 10:
            break
        perm = _neh_heuristic(pt, v)
        ms = _makespan(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
        
        # 2. Local Search on each variant
        if time.time() >= deadline - 5:
            break
        ls_perm = _local_search_insert(perm, pt, max_iters=20, deadline=deadline)
        ls_ms = _makespan(ls_perm, pt)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_perm = ls_perm
            
    # 3. Simulated Annealing on the best result
    if best_perm is not None and time.time() < deadline - 5:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, max_iters=1000)
        sa_ms = _makespan(sa_perm, pt)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_perm
            
    return list(best_perm)
