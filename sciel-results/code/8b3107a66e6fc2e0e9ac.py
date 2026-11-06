"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


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


def _calc_makespan_with_prefix(processing_times, perm, prefix_len):
    """
    Calculate makespan, returning both the makespan and completion array.
    If prefix_len is given, only recompute from prefix_len onward (assumes
    completion state at prefix_len is correct for the current perm).
    Returns (makespan, completion_list).
    """
    n_machines = processing_times.shape[1]
    
    if prefix_len is None or prefix_len <= 0:
        completion = [0.0] * n_machines
        start = 0
    else:
        # Recompute from scratch up to prefix_len to get correct state
        completion = [0.0] * n_machines
        for job in perm[:prefix_len]:
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
        start = prefix_len
    
    for idx in range(start, len(perm)):
        row = processing_times[perm[idx]]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1], completion


def _neh_heuristic(processing_times):
    """
    NEH heuristic with incremental evaluation for insertion.
    """
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            # Build candidate sequence
            if pos == 0:
                cand = [job] + sequence
            elif pos == len(sequence):
                cand = sequence + [job]
            else:
                cand = sequence[:pos] + [job] + sequence[pos:]
            
            ms = _calc_makespan(processing_times, cand)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Insertion local search with incremental makespan evaluation.
    For a move (remove at i, insert at j), only positions between the
    removal and insertion points change, so we can reuse prefix state.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Remove job at i, insert at position j (adjusted)
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                # Build new sequence
                seq_copy = current_seq[:]
                seq_copy.pop(i)
                seq_copy.insert(insert_idx, job)
                
                # Incremental evaluation:
                # If j > i: prefix is seq_copy[:i], suffix changes from i to insert_idx
                # If j <= i: prefix is seq_copy[:insert_idx], suffix changes from insert_idx to i
                if j > i:
                    prefix_len = i
                else:
                    prefix_len = insert_idx
                
                new_makespan, _ = _calc_makespan_with_prefix(
                    processing_times, seq_copy, prefix_len
                )
                
                if new_makespan < current_makespan:
                    current_seq = seq_copy
                    current_makespan = new_makespan
                    improved = True
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing to escape local minima.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if n_jobs < 2:
            break
        i, j = random.sample(range(n_jobs), 2)
        new_seq = current_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
        if T < 1e-6:
            break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Local search with time limit
    best_seq, best_makespan = _local_search(processing_times, initial_seq, deadline)
    
    # Step 3: Simulated Annealing if time permits
    if time.time() < deadline - 5.0:
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Step 4: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
    
    return best_seq


# EVOLVE-BLOCK-END
