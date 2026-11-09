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
    """
    Compute makespan and return the full completion time matrix C.
    C[i][j] is the completion time of job i on machine j.
    """
    if len(order) == 0:
        return 0.0, []
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

def _delta_swap(processing_times, order, C, i, j):
    """
    Calculate the change in makespan when swapping jobs at positions i and j.
    Assumes i < j.
    """
    if i == j:
        return 0.0
    
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Copy the completion time matrix
    # We only need to update rows i to j and potentially beyond if the makespan changes
    # But makespan is C[n_jobs-1][n_machines-1].
    # The swap affects C[i..j][0..n_machines-1] and potentially C[j+1..][0..n_machines-1]
    # Actually, C[k][m] for k > j depends on C[k-1][m] and C[k][m-1].
    # So we need to recalculate C[i..n_jobs-1][0..n_machines-1].
    
    # Optimization: We can just recalculate the affected part.
    # Let's create a shallow copy of the rows we need to update.
    # Since C is a list of lists, we can copy the rows.
    new_C = [row[:] for row in C]
    
    # Recalculate C[i..n_jobs-1]
    # We need to be careful with dependencies.
    # C[i][j] depends on C[i-1][j] and C[i][j-1].
    # If we swap jobs at i and j, the processing times for those jobs change.
    # p_i = processing_times[order[i]]
    # p_j = processing_times[order[j]]
    
    # For k in [i, j]:
    # The job at position k is now:
    # if k == i: job_j
    # if k == j: job_i
    # else: job_k
    
    # Let's iterate k from i to n_jobs-1
    for k in range(i, n_jobs):
        # Determine which job is at position k in the new order
        if k == i:
            job_idx = order[j]
        elif k == j:
            job_idx = order[i]
        else:
            job_idx = order[k]
        
        p_row = processing_times[job_idx]
        
        for m in range(n_machines):
            p = p_row[m]
            if k == 0 and m == 0:
                new_C[k][m] = p
            elif k == 0:
                new_C[k][m] = new_C[k][m-1] + p
            elif m == 0:
                new_C[k][m] = new_C[k-1][m] + p
            else:
                new_C[k][m] = max(new_C[k-1][m], new_C[k][m-1]) + p
    
    old_ms = C[n_jobs-1][n_machines-1]
    new_ms = new_C[n_jobs-1][n_machines-1]
    return new_ms - old_ms, new_C

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position
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

def _local_search(processing_times, order, time_limit=10.0):
    """
    Local search with swap and 2-opt moves, strictly time-limited.
    Uses incremental evaluation for swaps to speed up the search.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms, best_C = _makespan_with_c(processing_times, best_order)
    start_time = time.time()
    
    # We will iterate until time runs out or no improvement is found in a full pass
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try swaps with incremental evaluation
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                delta, new_C = _delta_swap(processing_times, best_order, best_C, i, j)
                if delta < 0:
                    # Accept swap
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    best_ms += delta
                    best_C = new_C
                    improved = True
        
        if time.time() - start_time > time_limit:
            break
            
        # Try 2-opt (reverse) - Full evaluation is safer and 2-opt is less frequent
        # We can skip 2-opt if we are time-pressed, or use it sparingly
        # For now, let's keep it but be careful
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Reverse segment i to j
                best_order[i:j+1] = best_order[i:j+1][::-1]
                ms, C_new = _makespan_with_c(processing_times, best_order)
                if ms < best_ms:
                    best_ms = ms
                    best_C = C_new
                    improved = True
                else:
                    # Revert reverse
                    best_order[i:j+1] = best_order[i:j+1][::-1]
        
        if not improved:
            break
            
    return best_order, best_ms

def _insertion_polish(processing_times, order, time_limit=5.0):
    """
    Final polish using insertion moves.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms, best_C = _makespan_with_c(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job = best_order[i]
            # Try inserting job at all other positions
            for j in range(n_jobs):
                if i == j:
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                # Remove job from i
                # Insert job at j
                # This changes the sequence
                # We can just construct the new order and evaluate
                # Since n_jobs is likely small enough, this is fine
                new_order = best_order[:i] + best_order[i+1:]
                # Adjust insertion index if j > i
                insert_idx = j if j < i else j - 1
                new_order.insert(insert_idx, job)
                
                ms, C_new = _makespan_with_c(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_C = C_new
                    best_order = new_order
                    improved = True
                    break # Restart loop for this job? Or continue?
                    # Usually first improvement is enough for polish
                    # Let's break and restart the outer loop to be safe
            if improved:
                break # Restart outer loop
        
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by time-limited local search.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 120s (contract says 120s, error says 150s, 
    # but contract says "finish within 120 s". Let's use 110s to be safe).
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    initial_ms = _makespan(processing_times, initial_order)
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Allocate most time to the first local search
    time_limit = remaining_time - 5.0  # Keep 5s safety
    best_order, best_ms = _local_search(
        processing_times, initial_order, 
        time_limit=time_limit
    )
    
    # If we have significant time left, try a few restarts
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        best_overall = best_order
        best_overall_ms = best_ms
        
        # Try a few random restarts with short time limits
        num_restarts = min(int(remaining_time / 10.0), 5)
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Random initial order
            random_order = list(range(n_jobs))
            random.shuffle(random_order)
            
            # Quick local search with shorter time limit
            tl = min(remaining_time - 5.0, 10.0)
            if tl <= 0:
                break
            order, ms = _local_search(
                processing_times, random_order,
                time_limit=tl
            )
            
            if ms < best_overall_ms:
                best_overall = order
                best_overall_ms = ms
            
            remaining_time = deadline - time.time()
            if remaining_time <= 5.0:
                break
        
        # Final polish
        if remaining_time > 2.0:
            best_overall, best_overall_ms = _insertion_polish(
                processing_times, best_overall,
                time_limit=min(remaining_time - 1.0, 5.0)
            )
        
        return best_overall
    
    # Final polish
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_order, best_ms = _insertion_polish(
            processing_times, best_order,
            time_limit=min(remaining_time - 1.0, 5.0)
        )
    
    return best_order
# EVOLVE-BLOCK-END
