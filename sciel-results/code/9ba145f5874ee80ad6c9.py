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
    
    # c_prev[j] is completion time of previous job on machine j
    # c_curr[j] is completion time of current job on machine j
    c_prev = np.zeros(n_machines)
    c_curr = np.zeros(n_machines)
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[n_machines - 1]


def _neh_rule(processing_times):
    """NEH heuristic for flow-shop problem."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    job_order = np.argsort(-totals)
    
    # Build sequence iteratively
    seq = []
    remaining = list(job_order)
    
    for _ in range(n_jobs):
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting the next job at each possible position
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [remaining[0]] + seq[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        seq = seq[:best_pos] + [remaining[0]] + seq[best_pos:]
        remaining.pop(0)
    
    return seq


def _local_search(processing_times, order, time_limit):
    """Swap-based local search with 2-opt and 3-opt moves."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        
        # Check time limit
        if time.time() - start_time > time_limit:
            break
        
        # Try all swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                # Check time limit periodically
                if time.time() - start_time > time_limit:
                    return best_order, best_ms
                
                # Try swap
                candidate = list(best_order)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(processing_times, candidate)
                
                if ms < best_ms - 1e-9:
                    best_order = candidate
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
        
        # If no swap improved, try 2-opt (reverse segment)
        if not improved:
            for i in range(n_jobs):
                for j in range(i + 2, n_jobs):
                    if time.time() - start_time > time_limit:
                        return best_order, best_ms
                    
                    candidate = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                    ms = _makespan(processing_times, candidate)
                    
                    if ms < best_ms - 1e-9:
                        best_order = candidate
                        best_ms = ms
                        improved = True
                        break
                if improved:
                    break
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    # Generate initial solution using NEH
    best_order = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_order)
    
    # Perform local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        best_order, best_ms = _local_search(processing_times, best_order, remaining_time - 1.0)
    
    # Final check: ensure we return within time limit
    if time.time() > deadline - 0.5:
        return list(best_order)
    
    return list(best_order)
# EVOLVE-BLOCK-END
