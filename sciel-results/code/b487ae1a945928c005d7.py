"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _completion_vector(processing_times, order):
    """Compute the completion time vector c for a given job order."""
    if len(order) == 0:
        return np.zeros(processing_times.shape[1])
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c

def _compute_c_prefix(processing_times, order):
    """Compute prefix completion vectors for all positions in order."""
    n_machines = processing_times.shape[1]
    n = len(order)
    c_prefix = [np.zeros(n_machines) for _ in range(n + 1)]
    for i in range(n):
        c_prefix[i + 1] = c_prefix[i].copy()
        p_row = processing_times[order[i]]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c_prefix[i + 1][j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c_prefix[i + 1][j] = c_val
            prev_c = c_val
    return c_prefix

def _makespan_from_suffix(processing_times, order, start_pos, c_start):
    """Compute makespan by processing order[start_pos:] starting from c_start."""
    n_machines = processing_times.shape[1]
    c = c_start.copy()
    for i in range(start_pos, len(order)):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _neh_construct(processing_times, sort_key='sum'):
    """
    NEH-style constructive heuristic.
    sort_key: 'sum', 'mean', 'min', 'max'
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    if sort_key == 'sum':
        totals = processing_times.sum(axis=1)
    elif sort_key == 'mean':
        totals = processing_times.mean(axis=1)
    elif sort_key == 'min':
        totals = processing_times.min(axis=1)
    elif sort_key == 'max':
        totals = processing_times.max(axis=1)
    else:
        totals = processing_times.sum(axis=1)
    
    sorted_jobs = list(np.argsort(-totals))
    
    current_order = []
    
    for job in sorted_jobs:
        if not current_order:
            current_order = [job]
            continue
        
        best_pos = len(current_order)
        best_ms = float('inf')
        
        # Precompute prefix completion vectors for current_order
        c_prefixes = _compute_c_prefix(processing_times, current_order)
        
        for pos in range(len(current_order) + 1):
            c_before = c_prefixes[pos]
            
            # Process the new job
            p_row = processing_times[job]
            c_after_job = c_before.copy()
            prev_c = 0.0
            for j in range(n_machines):
                c_val = c_after_job[j]
                if prev_c > c_val:
                    c_val = prev_c
                c_val += p_row[j]
                c_after_job[j] = c_val
                prev_c = c_val
            
            # Process remaining jobs
            ms = _makespan_from_suffix(processing_times, current_order, pos, c_after_job)
            
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        current_order.insert(best_pos, job)
    
    return current_order

def _local_search_insertion(processing_times, initial_order, deadline):
    """
    Local search using insertion moves (remove and reinsert).
    Uses prefix completion vectors for efficient evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(initial_order)
    
    best_order = list(initial_order)
    best_ms = _makespan(processing_times, best_order)
    
    improved = True
    max_ls_iter = 100
    ls_iter = 0
    
    while improved and time.time() < deadline and ls_iter < max_ls_iter:
        improved = False
        ls_iter += 1
        n = len(best_order)
        
        # Precompute prefix completion vectors for the current best order
        c_prefixes = _compute_c_prefix(processing_times, best_order)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            job = best_order[i]
            # Remove job from position i
            temp_order = best_order[:i] + best_order[i+1:]
            
            # Precompute prefix completion vectors for temp_order
            c_prefixes_temp = _compute_c_prefix(processing_times, temp_order)
            
            best_pos_local = i
            best_ms_local = float('inf')
            
            for pos in range(len(temp_order) + 1):
                # Compute makespan with job inserted at pos
                c_before = c_prefixes_temp[pos]
                
                # Process the new job
                p_row = processing_times[job]
                c_after_job = c_before.copy()
                prev_c = 0.0
                for j in range(n_machines):
                    c_val = c_after_job[j]
                    if prev_c > c_val:
                        c_val = prev_c
                    c_val += p_row[j]
                    c_after_job[j] = c_val
                    prev_c = c_val
                
                # Process remaining jobs
                ms = _makespan_from_suffix(processing_times, temp_order, pos, c_after_job)
                
                if ms < best_ms_local:
                    best_ms_local = ms
                    best_pos_local = pos
            
            if best_ms_local < best_ms - 1e-9:
                # Apply the best move
                best_order = temp_order[:best_pos_local] + [job] + temp_order[best_pos_local:]
                best_ms = best_ms_local
                improved = True
                # Restart the scan
                break
    
    return best_order

def _local_search_2opt(processing_times, initial_order, deadline):
    """
    Local search using 2-opt moves (reverse subsequence).
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 2:
        return list(initial_order)
    
    best_order = list(initial_order)
    best_ms = _makespan(processing_times, best_order)
    
    improved = True
    max_iters = 50
    iters = 0
    
    while improved and time.time() < deadline and iters < max_iters:
        improved = False
        iters += 1
        n = len(best_order)
        
        for i in range(n - 1):
            if time.time() > deadline:
                return best_order
            for j in range(i + 1, n):
                new_order = best_order.copy()
                new_order[i:j+1] = new_order[i:j+1][::-1]
                new_ms = _makespan(processing_times, new_order)
                if new_ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = new_ms
                    improved = True
    
    return best_order

def _simulated_annealing(processing_times, initial_order, deadline):
    """
    Simulated annealing with swap moves.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs < 2:
        return list(initial_order)
    
    current = list(initial_order)
    current_ms = _makespan(processing_times, current)
    best_order = list(current)
    best_ms = current_ms
    
    T = max(current_ms * 0.01, 1.0)
    cooling = 0.999
    min_T = max(current_ms * 1e-6, 1e-3)
    
    while T > min_T and time.time() < deadline:
        n = len(current)
        if n < 2:
            break
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            T *= cooling
            continue
        
        # Swap move
        new_order = current.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan(processing_times, new_order)
        delta = new_ms - current_ms
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current)
        
        T *= cooling
    
    return best_order

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 147  # 3s safety margin
    
    # Generate multiple starting solutions using different NEH variants
    candidates = []
    
    # 1. NEH with sum
    candidates.append(_neh_construct(processing_times, 'sum'))
    
    # 2. NEH with mean
    candidates.append(_neh_construct(processing_times, 'mean'))
    
    # 3. NEH with min
    candidates.append(_neh_construct(processing_times, 'min'))
    
    # 4. NEH with max
    candidates.append(_neh_construct(processing_times, 'max'))
    
    # 5. Random restarts
    rng = np.random.RandomState(42)
    for _ in range(3):
        random_order = list(rng.permutation(n_jobs))
        candidates.append(random_order)
    
    # Apply local search to each candidate and keep the best
    best_order = None
    best_ms = float('inf')
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # First apply insertion local search
        ls_order = _local_search_insertion(processing_times, candidate, deadline)
        ms = _makespan(processing_times, ls_order)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(ls_order)
    
    # If we have time, apply 2-opt local search to the best solution
    if best_order is not None and time.time() < deadline:
        best_order = _local_search_2opt(processing_times, best_order, deadline)
        best_ms = _makespan(processing_times, best_order)
    
    # If we still have time, apply simulated annealing
    if best_order is not None and time.time() < deadline:
        best_order = _simulated_annealing(processing_times, best_order, deadline)
    
    return best_order if best_order is not None else list(range(n_jobs))
# EVOLVE-BLOCK-END
