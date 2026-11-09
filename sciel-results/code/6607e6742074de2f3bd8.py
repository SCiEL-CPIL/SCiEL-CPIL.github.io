import numpy as np
import time
from itertools import permutations


def _makespan_fast(perm, pt):
    """Faster makespan using numpy where possible."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 1:
        return pt[perm, 0].sum()
    
    # Build the submatrix for the permutation
    sub = pt[perm, :]  # shape (n, m)
    
    # completion[i][j] = max(completion[i-1][j], completion[i][j-1]) + sub[i][j]
    # We can do this iteratively
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        # Vectorize the inner loop if possible?
        # comp[i, 1:] = max(comp[i-1, 1:], comp[i, :-1] + sub[i, 1:]) + sub[i, 1:]
        # This is not quite right because comp[i, j-1] depends on the previous step in the vector.
        # But we can use a loop for j, which is small (m).
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    
    return comp[-1, -1]

def _neh_order(pt):
    """NEH heuristic."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with first job
    perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _brute_force(pt):
    n = pt.shape[0]
    if n > 10:
        return None
    best_perm = None
    best_ms = float('inf')
    for perm in permutations(range(n)):
        ms = _makespan_fast(list(perm), pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    return best_perm

def _local_search_insertion(perm, pt, time_limit=10.0):
    """Local search using insertion moves."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_limit:
                break
            job = best_perm[i]
            rest = best_perm[:i] + best_perm[i+1:]
            for j in range(n):
                if j == i or (j > 0 and j == i + 1):
                    continue
                new_perm = rest[:j] + [job] + rest[j:]
                ms = _makespan_fast(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    i = -1
                    break
    return best_perm

def _simulated_annealing_insertion(perm, pt, time_limit=5.0):
    """Simulated annealing with insertion moves."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.99
    
    rng = np.random.default_rng(42)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        i = rng.integers(0, n)
        job = current_perm[i]
        rest = current_perm[:i] + current_perm[i+1:]
        j = rng.integers(0, n)
        if j == i or (j > 0 and j == i + 1):
            continue
            
        new_perm = rest[:j] + [job] + rest[j:]
        new_ms = _makespan_fast(new_perm, pt)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        T *= cooling
    
    return best_perm

def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 1:
        return list(np.argsort(pt[:, 0]))
    
    # Brute force for small instances
    if n <= 10:
        return _brute_force(pt)
    
    deadline = 115.0
    start_time = time.time()
    
    candidates = []
    
    # NEH
    neh_perm = _neh_order(pt)
    candidates.append(neh_perm)
    
    # Palmer
    palmer_keys = np.zeros(n)
    for j in range(n):
        for k in range(m):
            palmer_keys[j] += (n - 2*k - 1) * pt[j, k]
    palmer_perm = list(np.argsort(palmer_keys))
    candidates.append(palmer_perm)
    
    # Random (multiple)
    rng = np.random.default_rng(123)
    for _ in range(5):
        candidates.append(list(rng.permutation(n)))
    
    # Pick best
    best_perm = candidates[0]
    best_ms = _makespan_fast(best_perm, pt)
    for perm in candidates[1:]:
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Local Search
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        ls_budget = min(40.0, deadline - 15.0 - elapsed)
        best_perm = _local_search_insertion(best_perm, pt, time_limit=ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
        
    # Simulated Annealing
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        sa_budget = min(10.0, deadline - 5.0 - elapsed)
        sa_perm = _simulated_annealing_insertion(best_perm, pt, time_limit=sa_budget)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
            
    # Final Polish (short LS)
    elapsed = time.time() - start_time
    if elapsed < deadline - 2.0:
        polish_budget = min(2.0, deadline - 2.0 - elapsed)
        best_perm = _local_search_insertion(best_perm, pt, time_limit=polish_budget)
        best_ms = _makespan_fast(best_perm, pt)

    return list(best_perm)
