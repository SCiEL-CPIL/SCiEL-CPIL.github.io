"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = c[k]
            if val < prev_c:
                val = prev_c
            val += row[k]
            c[k] = val
            prev_c = val
    return c[-1]

def _neh(processing_times):
    """NEH heuristic to generate initial solution."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals).tolist()
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial_perm = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm

def _insertion_local_search(perm, processing_times, max_passes=20):
    """Local search using insertion moves with early stopping."""
    n = len(perm)
    current_ms = _makespan_fast(perm, processing_times)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                new_perm = perm.copy()
                new_perm.pop(i)
                adj_j = j - 1 if i < j else j
                if adj_j < 0 or adj_j >= len(new_perm):
                    continue
                new_perm.insert(adj_j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return perm, current_ms

def _sa_insertion(perm, pt, T_init, T_min, alpha, deadline, best_perm, best_ms):
    """Simulated Annealing with insertion moves."""
    n = len(perm)
    current_perm = perm.copy()
    current_ms = _makespan_fast(current_perm, pt)
    
    if current_ms < best_ms:
        best_ms = current_ms
        best_perm = current_perm.copy()
    
    T = T_init
    stall_count = 0
    max_stall = 500
    
    while time.time() < deadline and T > T_min:
        # Generate insertion move: pick a job and move it to a random position
        i = random.randint(0, n - 1)
        job = current_perm[i]
        
        # Pick a new position different from current
        if n <= 1:
            break
        j = random.randint(0, n - 1)
        if j == i:
            continue
        
        new_perm = current_perm.copy()
        new_perm.pop(i)
        adj_j = j - 1 if i < j else j
        if adj_j < 0 or adj_j >= len(new_perm):
            continue
        new_perm.insert(adj_j, job)
        
        new_ms = _makespan_fast(new_perm, pt)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            stall_count = 0
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm.copy()
        else:
            stall_count += 1
        
        T *= alpha
        
        if stall_count > max_stall:
            break
    
    return current_perm, current_ms, best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Generate multiple diverse initial solutions using NEH with different orderings
    candidates = []
    
    # Standard NEH (descending total processing time)
    perm1 = _neh(processing_times)
    candidates.append(perm1)
    
    # NEH with ascending order (reverse)
    totals = processing_times.sum(axis=1)
    order_asc = np.argsort(totals).tolist()
    perm2 = []
    for job in order_asc:
        best_pos = 0
        best_ms = float('inf')
        for k in range(len(perm2) + 1):
            trial_perm = perm2[:k] + [job] + perm2[k:]
            ms = _makespan_fast(trial_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm2.insert(best_pos, job)
    candidates.append(perm2)
    
    # Random permutations
    for _ in range(3):
        random_perm = list(range(n))
        random.shuffle(random_perm)
        candidates.append(random_perm)
    
    # Polish each candidate and keep the best
    best_perm = None
    best_ms = float('inf')
    
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        polished, ms = _insertion_local_search(cand, processing_times, max_passes=5)
        if ms < best_ms:
            best_ms = ms
            best_perm = polished
    
    # 2. Multi-restart SA with insertion moves
    random.seed(42)
    pt = processing_times
    
    T_init = best_ms * 0.1
    T_min = best_ms * 0.001
    alpha = 0.998
    
    n_restarts = 8
    time_per_restart = (deadline - time.time() - 15.0) / max(1, n_restarts)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # For restart 0, use the best found so far
        if restart == 0:
            current_perm = best_perm.copy()
        else:
            # Perturb the best solution with random insertions
            current_perm = best_perm.copy()
            n_perturb = min(n, random.randint(3, max(3, n // 3)))
            for _ in range(n_perturb):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                job = current_perm[i]
                current_perm.pop(i)
                adj_j = j - 1 if i < j else j
                if 0 <= adj_j < len(current_perm):
                    current_perm.insert(adj_j, job)
        
        # Run SA for this restart
        restart_deadline = min(deadline - 15.0, time.time() + time_per_restart)
        current_perm, current_ms, best_perm, best_ms = _sa_insertion(
            current_perm, pt, T_init, T_min, alpha, restart_deadline, best_perm, best_ms
        )
        
        # Polish the best found in this restart
        if time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion_local_search(best_perm.copy(), pt, max_passes=10)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish
    if time.time() < deadline - 5.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm.copy(), pt, max_passes=15)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
