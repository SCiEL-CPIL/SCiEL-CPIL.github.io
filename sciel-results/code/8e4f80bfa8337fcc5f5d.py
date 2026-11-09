"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(perm, pt):
    """Compute makespan using two-pass recurrence for speed."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return 0.0
    
    # First machine: cumulative sum
    prev = np.zeros(n)
    s = 0.0
    for i in range(n):
        s += pt[perm[i], 0]
        prev[i] = s
    
    # Subsequent machines
    for k in range(1, m):
        curr = np.zeros(n)
        curr[0] = prev[0] + pt[perm[0], k]
        for i in range(1, n):
            v1 = prev[i]
            v2 = curr[i-1]
            curr[i] = (v1 if v1 > v2 else v2) + pt[perm[i], k]
        prev = curr
    
    return prev[-1]


def _neh_constructive(pt, sort_key='sum'):
    """NEH heuristic: sort by key, insert each job at best position."""
    n, m = pt.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    if sort_key == 'sum':
        totals = pt.sum(axis=1)
    elif sort_key == 'mean':
        totals = pt.mean(axis=1)
    elif sort_key == 'min':
        totals = pt.min(axis=1)
    elif sort_key == 'max':
        totals = pt.max(axis=1)
    else:
        totals = pt.sum(axis=1)
    
    sorted_jobs = np.argsort(-totals).tolist()
    perm = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _local_search_insertion(perm, pt, deadline, max_passes=50):
    """Iterative improvement with insertion moves."""
    n = len(perm)
    if n <= 2:
        return perm
    
    best = list(perm)
    best_ms = _makespan_fast(best, pt)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = best[i]
            temp = best[:i] + best[i+1:]
            for j in range(len(temp) + 1):
                if j == i:
                    continue
                new_order = temp[:j] + [job] + temp[j:]
                ms = _makespan_fast(new_order, pt)
                if ms < best_ms - 1e-10:
                    best = new_order
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best


def _simulated_annealing(perm, pt, deadline, t0=1.0, cooling=0.9995, t_min=1e-3):
    """Simulated annealing with insertion moves."""
    n = len(perm)
    if n <= 2:
        return perm
    
    current = list(perm)
    current_ms = _makespan_fast(current, pt)
    best = list(current)
    best_ms = current_ms
    T = t0
    
    while T > t_min and time.time() < deadline:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            T *= cooling
            continue
        
        job = current[i]
        temp = current[:i] + current[i+1:]
        adj_j = j - 1 if i < j else j
        new_order = temp[:adj_j] + [job] + temp[adj_j:]
        
        new_ms = _makespan_fast(new_order, pt)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best = list(current)
        
        T *= cooling
    
    return best


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Phase 1: Generate diverse initial solutions
    candidates = []
    
    # NEH with sum
    candidates.append(_neh_constructive(processing_times, 'sum'))
    
    # NEH with mean
    candidates.append(_neh_constructive(processing_times, 'mean'))
    
    # LPT
    candidates.append(list(np.argsort(-processing_times.sum(axis=1))))
    
    # SPT
    candidates.append(list(np.argsort(processing_times.sum(axis=1))))
    
    # A few random restarts
    rng = np.random.RandomState(42)
    for _ in range(3):
        candidates.append(list(rng.permutation(n_jobs)))
    
    # Phase 2: Local search each candidate, keep best
    best_order = None
    best_ms = float('inf')
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        ls_order = _local_search_insertion(candidate, processing_times, deadline)
        ms = _makespan_fast(ls_order, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_order = ls_order
    
    if best_order is None:
        best_order = list(range(n_jobs))
        best_ms = _makespan_fast(best_order, processing_times)
    
    # Phase 3: Simulated annealing from best solution
    if time.time() < deadline:
        sa_order = _simulated_annealing(best_order, processing_times, deadline)
        sa_ms = _makespan_fast(sa_order, processing_times)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order
    
    # Phase 4: Final polish - another round of local search
    if time.time() < deadline - 5:
        polished = _local_search_insertion(best_order, processing_times, deadline, max_passes=20)
        polished_ms = _makespan_fast(polished, processing_times)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_order = polished
    
    return best_order
# EVOLVE-BLOCK-END
