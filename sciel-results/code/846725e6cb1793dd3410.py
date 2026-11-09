"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    c_prev = [0.0] * n_machines
    c_curr = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[-1]

def _makespan_with_c(processing_times, order):
    """Compute makespan and return the full C matrix for incremental evaluation."""
    if len(order) == 0:
        return 0.0, None
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    C = [[0.0] * n_machines for _ in range(n_jobs)]
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                C[i][j] = p
            elif i == 0:
                C[i][j] = C[i][j-1] + p
            elif j == 0:
                C[i][j] = C[i-1][j] + p
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + p
    
    return C[n_jobs-1][n_machines-1], C

def _delta_swap(processing_times, C, order, i, j):
    """
    Compute delta makespan for swapping positions i and j.
    Only recompute rows from min(i,j) to max(i,j).
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return 0.0
    
    lo, hi = min(i, j), max(i, j)
    
    # Build new C for affected rows
    # We need C[lo-1] as starting point (or zeros if lo==0)
    C_new = [row[:] for row in C]
    
    # Reconstruct rows from lo to hi with swapped jobs
    # Save original jobs at positions lo and hi
    job_lo = order[lo]
    job_hi = order[hi]
    
    # Swap in order temporarily for reference
    order[lo], order[hi] = order[hi], order[lo]
    
    for row in range(lo, hi + 1):
        job_idx = order[row]
        p_row = processing_times[job_idx]
        for col in range(n_machines):
            p = p_row[col]
            if row == 0 and col == 0:
                C_new[row][col] = p
            elif row == 0:
                C_new[row][col] = C_new[row][col-1] + p
            elif col == 0:
                C_new[row][col] = C_new[row-1][col] + p
            else:
                C_new[row][col] = max(C_new[row-1][col], C_new[row][col-1]) + p
    
    # Revert swap
    order[lo], order[hi] = order[hi], order[lo]
    
    new_ms = C_new[n_jobs-1][n_machines-1]
    old_ms = C[n_jobs-1][n_machines-1]
    
    return new_ms - old_ms

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _local_search_swaps(processing_times, order, time_limit=10.0):
    """
    Local search using swap moves.
    Iterates through all pairs and swaps if it improves the makespan.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                return current_order, current_ms
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    return current_order, current_ms
                
                # Swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_ms = _makespan(processing_times, current_order)
                
                if new_ms < current_ms:
                    current_ms = new_ms
                    improved = True
                else:
                    # Revert
                    current_order[i], current_order[j] = current_order[j], current_order[i]
    
    return current_order, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by time-limited tabu search with restarts.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Use most of the remaining time for local search
    time_limit = remaining_time * 0.9
    best_order, best_ms = _local_search_swaps(
        processing_times, initial_order, 
        time_limit=time_limit
    )
    
    return best_order
# EVOLVE-BLOCK-END
