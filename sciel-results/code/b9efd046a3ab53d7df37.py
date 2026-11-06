# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Compute cumulative completion times
    # For a flow shop, the completion time of job j on machine k is:
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + P[j][k]
    # where C[-1][k] = 0 and C[j][-1] = 0
    
    # We can compute this efficiently
    cum = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job_idx = order[j]
        for k in range(n_machines):
            if j == 0:
                prev_job_time = 0.0
            else:
                prev_job_time = cum[j-1, k]
            if k == 0:
                prev_machine_time = 0.0
            else:
                prev_machine_time = cum[j, k-1]
            cum[j, k] = max(prev_job_time, prev_machine_time) + processing_times[job_idx, k]
    
    return cum[-1, -1]


def _makespan_fast(processing_times, order):
    """Faster makespan computation using numpy."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Extract the submatrix for the given order
    sub = processing_times[order]
    
    # Use dynamic programming approach
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    # C[-1][j] = 0, C[i][-1] = 0
    
    # We'll build the cumulative table
    cum = np.zeros((n_jobs, n_machines))
    for i in range(n_jobs):
        for j in range(n_machines):
            up = cum[i-1, j] if i > 0 else 0.0
            left = cum[i, j-1] if j > 0 else 0.0
            cum[i, j] = max(up, left) + sub[i, j]
    
    return cum[-1, -1]


def _neh_initialization(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Sort jobs by total processing time in non-increasing order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Build solution by inserting each job in the best position
    order = [sorted_jobs[0]]
    
    for idx in range(1, n_jobs):
        job = sorted_jobs[idx]
        # Try inserting at each position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan_fast(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _adjacent_swap_delta(processing_times, order, i):
    """Compute makespan change when swapping positions i and i+1."""
    n_jobs = len(order)
    if i < 0 or i >= n_jobs - 1:
        return float('inf')
    
    # Compute makespan before and after swap
    ms_before = _makespan_fast(processing_times, order)
    order_swapped = order[:]
    order_swapped[i], order_swapped[i+1] = order_swapped[i+1], order_swapped[i]
    ms_after = _makespan_fast(processing_times, order_swapped)
    
    return ms_after - ms_before


def _improve_with_local_search(processing_times, order, time_limit, start_time):
    """Improve solution using local search with time limit."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return order
    
    current_order = order[:]
    current_ms = _makespan_fast(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
        
        for i in range(n_jobs - 1):
            # Check time
            if time.time() - start_time > time_limit - 5:
                break
            
            order_new = current_order[:]
            order_new[i], order_new[i+1] = order_new[i+1], order_new[i]
            new_ms = _makespan_fast(processing_times, order_new)
            
            if new_ms < current_ms:
                current_order = order_new
                current_ms = new_ms
                improved = True
                # Restart scan from beginning after improvement
                break
    
    return current_order


def _multi_start_local_search(processing_times, n_starts, time_per_start):
    """Multi-start local search with different initializations."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    best_order = None
    best_ms = float('inf')
    
    # Start 1: NEH
    start_time = time.time()
    order_neh = _neh_initialization(processing_times)
    order_improved = _improve_with_local_search(
        processing_times, order_neh, time_per_start, start_time
    )
    ms = _makespan_fast(processing_times, order_improved)
    if ms < best_ms:
        best_ms = ms
        best_order = order_improved
    
    # Additional starts with random perturbations of NEH
    for start in range(1, n_starts):
        if time.time() - start_time > time_per_start * (start + 1) - 10:
            break
        
        # Perturb the NEH solution
        order_perturbed = order_neh[:]
        # Do a few random swaps
        n_swaps = max(1, n_jobs // 5)
        for _ in range(n_swaps):
            i = np.random.randint(0, n_jobs - 1)
            order_perturbed[i], order_perturbed[i+1] = order_perturbed[i+1], order_perturbed[i]
        
        start_time_sub = time.time()
        order_improved = _improve_with_local_search(
            processing_times, order_perturbed, time_per_start, start_time_sub
        )
        ms = _makespan_fast(processing_times, order_improved)
        if ms < best_ms:
            best_ms = ms
            best_order = order_improved
    
    return best_order if best_order is not None else order_neh


def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    processing_times = np.asarray(processing_times)
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # For small problems, use more thorough search
    if n_jobs <= 20:
        # Use multi-start with more iterations
        n_starts = 10
        time_per_start = min(10.0, (deadline - (time.time() - start_time)) / n_starts)
        result = _multi_start_local_search(processing_times, n_starts, time_per_start)
    else:
        # For larger problems, be more conservative
        n_starts = 5
        time_per_start = min(5.0, (deadline - (time.time() - start_time)) / n_starts)
        result = _multi_start_local_search(processing_times, n_starts, time_per_start)
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > deadline - 2:
        # Return the best we have
        pass
    
    return result
# EVOLVE-BLOCK-END
