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
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
    return c[-1]

def _neh(processing_times):
    """NEH heuristic: build solution by inserting jobs in order of decreasing total processing time."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n)), 0.0
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = -1
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm, _makespan_fast(perm, processing_times)

def _insertion_local_search(perm, pt, max_iters=500):
    """
    Insertion-based local search: try moving each job to every other position.
    Returns improved permutation and makespan.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            # Try inserting at each position
            for k in range(n - 1):
                if k == i:
                    continue
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                if ms < current_ms:
                    current_perm = trial
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def _tabu_search(perm, pt, max_iters=2000, tabu_len=10, deadline=None):
    """
    Tabu search with aspiration criterion for flow-shop problem.
    Uses insertion moves with a tabu list to escape local optima.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list stores (job, position) pairs
    tabu_list = []
    tabu_tenure = max(5, n // 3)
    
    # Track best solution for aspiration
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        best_move_ms = float('inf')
        best_move_job = -1
        best_move_from = -1
        best_move_to = -1
        is_aspiration = False
        
        # Evaluate all insertion moves
        for i in range(n):
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            for k in range(n - 1):
                if k == i:
                    continue
                
                # Check tabu status
                is_tabu = False
                for (t_job, t_pos) in tabu_list:
                    if t_job == job and t_pos == k:
                        is_tabu = True
                        break
                
                if is_tabu:
                    # Aspiration criterion: allow if it improves best known
                    trial = without_job[:k] + [job] + without_job[k:]
                    ms = _makespan_fast(trial, pt)
                    if ms < best_ms:
                        is_aspiration = True
                    else:
                        continue
                else:
                    trial = without_job[:k] + [job] + without_job[k:]
                    ms = _makespan_fast(trial, pt)
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_job = job
                    best_move_from = i
                    best_move_to = k
                    is_aspiration = False
        
        if best_move_job == -1:
            break
        
        # Apply best move
        current_perm = current_perm[:best_move_from] + current_perm[best_move_from+1:]
        current_perm.insert(best_move_to, best_move_job)
        current_ms = best_move_ms
        
        # Update tabu list
        tabu_list.append((best_move_job, best_move_to))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Update best
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
    
    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
    
    # 2. Multi-restart with Tabu Search and Simulated Annealing
    rng = np.random.RandomState(42)
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        num_perturb = max(1, n // 8)
        for _ in range(num_perturb):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            job = current_perm[i]
            current_perm.pop(i)
            adj_j = j - 1 if i < j else j
            if 0 <= adj_j < len(current_perm):
                current_perm.insert(adj_j, job)
        
        # Alternate between Tabu Search and SA based on restart parity
        if restart % 2 == 0:
            # Tabu Search
            ts_deadline = min(deadline - 10.0, time.time() + 15.0)
            ts_perm, ts_ms = _tabu_search(current_perm, processing_times, max_iters=3000, deadline=ts_deadline)
            if ts_ms < best_ms:
                best_ms = ts_ms
                best_perm = ts_perm
        else:
            # Simulated Annealing with insertion moves
            current_ms = _makespan_fast(current_perm, processing_times)
            T = best_ms * 0.05
            T_min = best_ms * 0.001
            alpha = 0.9995
            
            stall_count = 0
            max_stall = 800
            
            while time.time() < deadline - 10.0:
                if T < T_min:
                    break
                
                # Generate insertion move
                i = rng.randint(0, n - 1)
                job = current_perm[i]
                j = rng.randint(0, n - 1)
                if j == i:
                    continue
                
                new_perm = current_perm[:]
                new_perm.pop(i)
                adj_j = j - 1 if i < j else j
                if 0 <= adj_j < len(new_perm):
                    new_perm.insert(adj_j, job)
                
                new_ms = _makespan_fast(new_perm, processing_times)
                delta = new_ms - current_ms
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_perm = new_perm
                    current_ms = new_ms
                    stall_count = 0
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                else:
                    stall_count += 1
                
                T *= alpha
                
                if stall_count > max_stall:
                    break
        
        # Apply insertion local search to polish
        if time.time() < deadline - 5.0:
            polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=30)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 2.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
