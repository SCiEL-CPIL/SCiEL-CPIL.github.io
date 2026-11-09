"""Improved program for flow-shop permutation construction using NEH + ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Optimized for speed using Python lists for completion times.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence to minimize makespan.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _local_search(order, P, time_limit, start_time):
    """
    Perform first-improvement insertion local search.
    Returns the best order and makespan found.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    current_order = order[:]
    current_makespan = _makespan(current_order, P)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Randomize the order of jobs to check to avoid bias
        job_indices = list(range(n_jobs))
        random.shuffle(job_indices)
        
        for i in job_indices:
            if (time.time() - start_time) > time_limit:
                break
                
            job_i = current_order[i]
            
            # Try moving job_i to all other positions
            for j in range(n_jobs):
                if i == j:
                    continue
                
                # Construct new order
                new_order = current_order[:]
                new_order.pop(i)
                new_order.insert(j, job_i)
                
                ms = _makespan(new_order, P)
                
                if ms < current_makespan:
                    current_makespan = ms
                    current_order = new_order
                    improved = True
                    break # Restart the loop with the new order
        
    return current_order, current_makespan

def _perturb(order, P, strength):
    """
    Perturb the order by re-inserting a random block of jobs.
    strength: number of jobs in the block.
    """
    n_jobs = len(order)
    if n_jobs < 2:
        return order[:]
    
    block_size = min(strength, n_jobs)
    start = random.randint(0, n_jobs - block_size)
    end = start + block_size
    
    block = order[start:end]
    remaining = order[:start] + order[end:]
    
    # Insert block back into a random position
    insert_pos = random.randint(0, len(remaining))
    new_order = remaining[:insert_pos] + block + remaining[insert_pos:]
    
    return new_order

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH heuristic followed by Iterated Local Search.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with NEH
    best_order = _neh_heuristic(P)
    best_makespan = _makespan(best_order, P)
    
    # 2. Initial Local Search
    best_order, best_makespan = _local_search(best_order, P, time_limit, start_time)
    
    # 3. Iterated Local Search
    # Use a simple acceptance criterion: accept if better, or with a small probability if worse (simulated annealing style)
    # But to keep it simple and robust, we'll just accept if better, and restart if stuck.
    
    current_order = best_order[:]
    current_makespan = best_makespan
    
    # Determine perturbation strength based on problem size
    # Smaller strength for larger problems to avoid too much disruption
    base_strength = max(2, int(np.sqrt(n_jobs)))
    
    no_improve_count = 0
    max_no_improve = 10
    
    while (time.time() - start_time) < time_limit:
        # Perturb
        strength = base_strength
        # Increase strength if stuck
        if no_improve_count > 5:
            strength = min(n_jobs, base_strength * 2)
            
        perturbed_order = _perturb(current_order, P, strength)
        
        # Local Search
        perturbed_order, perturbed_makespan = _local_search(perturbed_order, P, time_limit, start_time)
        
        # Accept if better
        if perturbed_makespan < best_makespan:
            best_makespan = perturbed_makespan
            best_order = perturbed_order
            current_order = perturbed_order
            current_makespan = perturbed_makespan
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Accept with small probability to escape local optima
            # Simple SA-like acceptance: accept if rand() < exp(-delta/T)
            # But we don't have a temperature. Let's just restart if stuck for a while.
            if no_improve_count > max_no_improve:
                # Restart from best
                current_order = best_order[:]
                current_makespan = best_makespan
                no_improve_count = 0
        
        # Check time
        if (time.time() - start_time) > time_limit:
            break
            
    return best_order
# EVOLVE-BLOCK-END
