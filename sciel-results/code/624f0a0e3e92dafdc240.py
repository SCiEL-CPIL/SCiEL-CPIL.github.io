"""Alternative strategy: NEH initialization + Simulated Annealing with Or-opt/2-opt moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
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

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH + Simulated Annealing.
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
    
    # Initialize with NEH
    current_order = _neh_heuristic(P)
    current_makespan = _makespan(current_order, P)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # Simulated Annealing Parameters
    T = 100.0  # Initial temperature
    cooling_rate = 0.99
    min_temperature = 0.1
    max_steps = 100000
    
    rng = random.Random(42)
    
    step = 0
    while (time.time() - start_time) < time_limit and T > min_temperature and step < max_steps:
        step += 1
        
        # Generate a neighbor move
        move_type = rng.choice(['swap', 'or_opt'])
        
        if move_type == 'swap' and n_jobs >= 2:
            i, j = rng.sample(range(n_jobs), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 'or_opt' and n_jobs >= 3:
            # Or-opt: move a block of 1, 2, or 3 jobs to a new position
            block_len = rng.choice([1, 2, 3])
            if n_jobs < block_len + 1:
                block_len = max(1, n_jobs - 1)
            
            i = rng.randint(0, n_jobs - block_len)
            block = current_order[i:i+block_len]
            
            # Remove block
            new_order = current_order[:i] + current_order[i+block_len:]
            
            # Insert block at random position (excluding original position to ensure a move)
            possible_positions = list(range(len(new_order) + 1))
            # The original position in the new list would be i. We want to move it elsewhere.
            # If i == 0, the block was at the start. In the new list, inserting at 0 puts it back.
            # We should avoid inserting at the position that restores the original order.
            # Original index of the block in the new list:
            # If we remove [i:i+L], the element at i moves to i-L (if i>0) or 0 (if i=0? No, indices shift).
            # Let's just pick a random position and if it results in the same order, skip or pick another.
            # Simpler: pick a random position j in [0, len(new_order)].
            j = rng.randint(0, len(new_order))
            
            # Check if this is a trivial move (inserting back at original spot)
            # If i == 0, original spot is 0.
            # If i > 0, original spot is i - block_len? No.
            # Let's just apply the move and check if new_order == current_order.
            new_order = new_order[:j] + block + new_order[j:]
            
            if new_order == current_order:
                continue
            
        else:
            continue
        
        new_makespan = _makespan(new_order, P)
        delta = new_makespan - current_makespan
        
        if delta < 0:
            # Accept improvement
            current_order = new_order
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_order = current_order[:]
                best_makespan = current_makespan
        else:
            # Accept worse solution with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / T):
                current_order = new_order
                current_makespan = new_makespan
        
        T *= cooling_rate
        
    return best_order
# EVOLVE-BLOCK-END
