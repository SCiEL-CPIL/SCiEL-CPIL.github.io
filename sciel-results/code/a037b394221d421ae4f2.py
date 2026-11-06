"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]


def _calc_makespan_fast(processing_times, perm):
    """
    Calculate makespan using numpy vectorization for speed.
    perm is a list or array of job indices.
    """
    n_machines = processing_times.shape[1]
    P = processing_times[perm]  # shape (n_jobs, n_machines)
    
    # C[j, m] is completion time of job j on machine m
    # C[0, 0] = P[0, 0]
    # C[0, m] = C[0, m-1] + P[0, m] for m > 0
    # C[j, 0] = C[j-1, 0] + P[j, 0]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    C = np.zeros((n_jobs, n_machines))
    
    # First job
    C[0, 0] = P[0, 0]
    if n_machines > 1:
        C[0, 1:] = np.cumsum(P[0, 1:])
    
    # Remaining jobs
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + P[j, 0]
        if n_machines > 1:
            # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
            # We can compute this iteratively
            for m in range(1, n_machines):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    return C[-1, -1]


def _neh_heuristic(processing_times):
    """
    NEH heuristic: 
    1. Sort jobs by total processing time descending.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with the first job
    sequence = [int(order[0])]
    
    for i in range(1, n_jobs):
        job = int(order[i])
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Or-opt local search: try moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs of positions (i, j)
        # Move job at i to position j
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Perform move
                seq_copy = current_seq[:]
                job_to_move = seq_copy.pop(i)
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                seq_copy.insert(insert_idx, job_to_move)
                
                new_makespan = _calc_makespan(processing_times, seq_copy)
                
                if new_makespan < current_makespan:
                    current_seq = seq_copy
                    current_makespan = new_makespan
                    improved = True
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def _adjacent_swap_search(processing_times, sequence, deadline):
    """
    Adjacent swap local search: try swapping adjacent jobs.
    This is a smaller neighborhood but faster per evaluation.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            
            # Swap i and i+1
            seq_copy = current_seq[:]
            seq_copy[i], seq_copy[i+1] = seq_copy[i+1], seq_copy[i]
            
            new_makespan = _calc_makespan(processing_times, seq_copy)
            
            if new_makespan < current_makespan:
                current_seq = seq_copy
                current_makespan = new_makespan
                improved = True
                # After a swap, the indices have shifted, so restart from beginning
                break
    
    return current_seq, current_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Or-opt local search (stronger neighborhood)
    best_seq, best_makespan = _local_search(processing_times, initial_seq, deadline)
    
    # Step 3: If time remains, do adjacent swap polish
    if time.time() < deadline:
        best_seq, best_makespan = _adjacent_swap_search(processing_times, best_seq, deadline)
    
    return best_seq


# EVOLVE-BLOCK-END
