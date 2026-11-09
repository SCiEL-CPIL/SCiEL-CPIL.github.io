"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order using the flow-shop recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] holds completion time on machine j for the last processed job
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]

def _neh_rule(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow-shop scheduling."""
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence

def _insertion_makespan_delta(processing_times, order, job, pos):
    """Compute the change in makespan when inserting job at position pos in order."""
    # This is a simplified version - for efficiency we recompute
    candidate = order[:pos] + [job] + order[pos:]
    return _makespan(processing_times, candidate)

def _local_search(processing_times, order, max_iterations=500, time_limit=None):
    """Insertion-based local search to improve the sequence."""
    n_jobs = len(order)
    best_order = list(order)
    best_makespan = _makespan(processing_times, best_order)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        # Try removing each job and reinserting it at the best position
        for i in range(n_jobs):
            job = best_order[i]
            # Remove job from position i
            remaining = best_order[:i] + best_order[i+1:]
            
            # Find best insertion position
            best_pos = 0
            best_ms = best_makespan
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue  # Skip original position
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < best_makespan:
                best_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                best_makespan = best_ms
                improved = True
                break  # Restart with improved sequence
        
        if not improved:
            break
    
    return best_order, best_makespan

def _simulated_annealing(processing_times, order, time_budget=30.0):
    """Simulated annealing to escape local optima."""
    n_jobs = len(order)
    current_order = list(order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    initial_temp = max(1.0, current_makespan * 0.1)
    cooling_rate = 0.995
    
    temp = initial_temp
    while temp > 0.01 and (time.time() - start_time) < time_budget:
        # Generate neighbor by swapping two adjacent jobs
        if n_jobs < 2:
            break
        i = random.randint(0, n_jobs - 2)
        neighbor = list(current_order)
        neighbor[i], neighbor[i+1] = neighbor[i+1], neighbor[i]
        
        neighbor_makespan = _makespan(processing_times, neighbor)
        delta = neighbor_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_order = neighbor
            current_makespan = neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
        
        temp *= cooling_rate
    
    return best_order, best_makespan

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic, insertion local search, and simulated annealing.
    """
    start_time = time.time()
    time_limit = 120.0  # Evaluator kills after 150s, use 120s for safety
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Step 1: Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    initial_makespan = _makespan(processing_times, initial_order)
    
    best_order = list(initial_order)
    best_makespan = initial_makespan
    
    # Step 2: Insertion-based local search
    ls_time_budget = min(20.0, time_limit * 0.2)
    ls_order, ls_makespan = _local_search(
        processing_times, initial_order, 
        max_iterations=1000, 
        time_limit=ls_time_budget
    )
    
    if ls_makespan < best_makespan:
        best_order = ls_order
        best_makespan = ls_makespan
    
    # Step 3: Simulated annealing with remaining time
    elapsed = time.time() - start_time
    sa_time_budget = max(1.0, time_limit - elapsed - 5.0)  # Leave 5s buffer
    
    if sa_time_budget > 1.0:
        sa_order, sa_makespan = _simulated_annealing(
            processing_times, best_order,
            time_budget=sa_time_budget
        )
        
        if sa_makespan < best_makespan:
            best_order = sa_order
            best_makespan = sa_makespan
    
    # Step 4: Final local search polish if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10.0:
        final_time_budget = time_limit - elapsed - 5.0
        if final_time_budget > 1.0:
            fs_order, fs_makespan = _local_search(
                processing_times, best_order,
                max_iterations=500,
                time_limit=final_time_budget
            )
            if fs_makespan < best_makespan:
                best_order = fs_order
                best_makespan = fs_makespan
    
    # Safety check: ensure we're within time limit
    if time.time() - start_time > time_limit - 2.0:
        # Return best found so far
        pass
    
    return best_order
# EVOLVE-BLOCK-END
