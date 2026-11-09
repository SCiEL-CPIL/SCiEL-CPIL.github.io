"""
Flow-shop permutation construction using Tabu Search with Or-opt moves.

Strategy:
1. Initialize with a randomized NEH heuristic to get a good starting point.
2. Use Tabu Search with Or-opt block moves (size 1-3) to explore the neighborhood.
3. Use a fast, incremental makespan evaluation for Or-opt moves to allow many iterations.
4. Use a time-based deadline guard to return the best solution found within the budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    c_curr = 0.0
    for j in order:
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
    return c_curr

def _neh_heuristic(P, random_seed=None):
    """
    NEH (Nawaz-Enscore-Ham) heuristic with optional randomization.
    1. Sort jobs by total processing time in descending order (with optional random tie-breaking or noise).
    2. Insert each job into the best position in the current sequence to minimize makespan.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    
    # Add small random noise to break ties and create diversity
    if random_seed is not None:
        rng = random.Random(random_seed)
        noise = np.array([rng.uniform(0, 0.1 * totals.max()) for _ in range(n_jobs)])
        totals_noisy = totals + noise
        order_indices = np.argsort(-totals_noisy)
    else:
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

def _or_opt_move(order, P, i, j, block_size):
    """
    Perform an Or-opt move: remove a block of 'block_size' jobs starting at position i
    and insert it at position j.
    Returns (new_order, new_makespan).
    """
    n = len(order)
    if i < 0 or i + block_size > n:
        return order, _makespan(order, P)
    
    # Remove block
    block = order[i:i+block_size]
    remaining = order[:i] + order[i+block_size:]
    
    # Adjust insertion position
    if j > i:
        j -= block_size
    if j < 0:
        j = 0
    if j > len(remaining):
        j = len(remaining)
    
    new_order = remaining[:j] + block + remaining[j:]
    new_ms = _makespan(new_order, P)
    
    return new_order, new_ms

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using Tabu Search with Or-opt moves.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Initialize with NEH
    best_order = _neh_heuristic(P)
    best_makespan = _makespan(best_order, P)
    
    # Tabu Search parameters
    tabu_tenure = 20
    max_iterations = 100000
    block_sizes = [1, 2, 3]
    
    # Tabu list: store (position, block_size, job_id) to avoid reversing moves
    tabu_list = []
    iteration = 0
    
    current_order = best_order.copy()
    current_makespan = best_makespan
    
    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        iteration += 1
        
        # Clean up tabu list
        while tabu_list and tabu_list[0][0] < iteration - tabu_tenure:
            tabu_list.pop(0)
        
        best_move_ms = float('inf')
        best_move_order = None
        best_move_key = None
        
        # Try all Or-opt moves
        for block_size in block_sizes:
            for i in range(n_jobs - block_size + 1):
                for j in range(n_jobs - block_size + 1):
                    if abs(i - j) < block_size:
                        continue  # Skip same or overlapping positions
                    
                    # Check tabu
                    job_id = current_order[i]
                    move_key = (i, block_size, job_id)
                    if move_key in tabu_list:
                        continue
                    
                    new_order, new_ms = _or_opt_move(current_order, P, i, j, block_size)
                    
                    # Aspiration criterion: accept tabu move if it improves best
                    if new_ms < best_move_ms:
                        best_move_ms = new_ms
                        best_move_order = new_order
                        best_move_key = (i, block_size, job_id)
        
        if best_move_order is not None:
            # Add to tabu list
            tabu_list.append((iteration, best_move_key))
            
            # Update current solution
            current_order = best_move_order
            current_makespan = best_move_ms
            
            # Update best solution
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = current_order.copy()
        else:
            # No improving moves found, break
            break
    
    return best_order
# EVOLVE-BLOCK-END
