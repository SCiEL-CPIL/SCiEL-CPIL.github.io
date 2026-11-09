"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

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

def _delta_insertion(processing_times, C, order, src, dst):
    """
    Compute delta makespan for moving job from position src to position dst.
    Only recompute affected rows.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if src == dst:
        return 0.0
    
    # Determine the range of rows to recompute
    if src < dst:
        lo = src
        hi = dst
    else:
        lo = src - 1
        hi = dst
    
    # Build the new order
    job_to_move = order[src]
    new_order = order[:]
    del new_order[src]
    if src < dst:
        new_order.insert(dst, job_to_move)
    else:
        new_order.insert(dst, job_to_move)
    
    # Recompute C from lo to hi
    C_new = [row[:] for row in C]
    
    for row in range(lo, hi + 1):
        job_idx = new_order[row]
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

def _insertion_local_search(processing_times, order, time_limit=10.0):
    """
    Local search using insertion moves.
    Tries all possible insertions and takes the best improving move.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms, C = _makespan_with_c(processing_times, current_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        best_delta = 0.0
        best_src = -1
        best_dst = -1
        
        for src in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for dst in range(n_jobs):
                if dst == src or dst == src + 1:
                    continue
                delta = _delta_insertion(processing_times, C, current_order, src, dst)
                if delta < best_delta:
                    best_delta = delta
                    best_src = src
                    best_dst = dst
        
        if best_src >= 0 and best_delta < 0:
            # Apply the best insertion
            job_to_move = current_order[best_src]
            del current_order[best_src]
            if best_src < best_dst:
                current_order.insert(best_dst, job_to_move)
            else:
                current_order.insert(best_dst, job_to_move)
            
            current_ms += best_delta
            C = _makespan_with_c(processing_times, current_order)[1]
            improved = True
        else:
            break
    
    return current_order, current_ms

def _simulated_annealing(processing_times, order, time_limit=10.0):
    """
    Simulated annealing with insertion moves to escape local optima.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    current_order = list(order)
    current_ms, C = _makespan_with_c(processing_times, current_order)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    start_time = time.time()
    t0 = time.time()
    initial_temp = max(current_ms * 0.01, 1.0)
    cooling_rate = 0.995
    
    temp = initial_temp
    no_improve_count = 0
    
    while (time.time() - start_time < time_limit) and temp > 0.01:
        # Generate random insertion move
        src = random.randint(0, n_jobs - 1)
        dst = random.randint(0, n_jobs - 1)
        if dst == src or dst == src + 1:
            continue
        
        delta = _delta_insertion(processing_times, C, current_order, src, dst)
        
        # Metropolis acceptance
        if delta < 0 or random.random() < math.exp(-delta / temp):
            # Apply move
            job_to_move = current_order[src]
            del current_order[src]
            if src < dst:
                current_order.insert(dst, job_to_move)
            else:
                current_order.insert(dst, job_to_move)
            
            current_ms += delta
            C = _makespan_with_c(processing_times, current_order)[1]
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        temp *= cooling_rate
        
        # Early termination if stuck for too long
        if no_improve_count > 200:
            break
    
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by insertion-based local search and simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    
    # Run insertion local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Phase 1: Insertion local search (40% of remaining time)
    ls_time = remaining_time * 0.4
    ls_order, ls_ms = _insertion_local_search(
        processing_times, initial_order, 
        time_limit=ls_time
    )
    
    remaining_time = deadline - time.time()
    if remaining_time < 3.0:
        return ls_order
    
    # Phase 2: Simulated annealing (remaining time minus safety margin)
    sa_time = remaining_time * 0.5
    best_order, best_ms = _simulated_annealing(
        processing_times, ls_order,
        time_limit=sa_time
    )
    
    # Phase 3: Final insertion local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        polish_time = remaining_time * 0.5
        final_order, final_ms = _insertion_local_search(
            processing_times, best_order,
            time_limit=polish_time
        )
        if final_ms < best_ms:
            best_order = final_order
            best_ms = final_ms
    
    return best_order
# EVOLVE-BLOCK-END
