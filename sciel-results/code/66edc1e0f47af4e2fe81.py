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


def _calc_makespan_suffix(processing_times, jobs, comp_start):
    """
    Calculate makespan for a sequence of jobs starting with a given completion state.
    Returns (final_makespan, final_comp).
    """
    n_machines = processing_times.shape[1]
    comp = list(comp_start)
    
    for job in jobs:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                comp[0] += row[m]
            else:
                c_prev = comp[m-1]
                c_curr = comp[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                comp[m] = c_curr + row[m]
    
    return comp[-1], comp


def _get_prefix_comp(processing_times, seq, k):
    """Get completion times after first k jobs in seq."""
    n_machines = processing_times.shape[1]
    comp = [0.0] * n_machines
    for i in range(k):
        job = seq[i]
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                comp[0] += row[m]
            else:
                c_prev = comp[m-1]
                c_curr = comp[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                comp[m] = c_curr + row[m]
    return comp


def _neh_heuristic(processing_times):
    """
    NEH heuristic.
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
            # Optimize: only compute suffix from pos
            # Prefix is sequence[:pos]
            # New job is inserted at pos
            # Suffix is sequence[pos:]
            
            # We need comp at pos-1
            if pos == 0:
                comp_start = [0.0] * n_machines
            else:
                comp_start = _get_prefix_comp(processing_times, sequence, pos)
            
            # The jobs to process are: [job] + sequence[pos:]
            # Note: sequence[pos:] is a slice, creates a new list. 
            # For efficiency, we can pass the original and indices, but let's keep it simple for now.
            # Actually, _calc_makespan_suffix expects a list of jobs.
            # We can construct the list: [job] + sequence[pos:]
            jobs_to_eval = [job] + sequence[pos:]
            
            ms, _ = _calc_makespan_suffix(processing_times, jobs_to_eval, comp_start)
            
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Insertion local search with incremental makespan evaluation.
    Includes 1-job and 2-job moves.
    """
    n_jobs = len(sequence)
    n_machines = processing_times.shape[1]
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1-job moves
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Determine new sequence and evaluate
                # If i < j: move seq[i] to j.
                # Prefix: seq[0:i]
                # Middle: seq[i+1:j+1]
                # Job: seq[i]
                # Suffix: seq[j+1:]
                
                if i < j:
                    prefix_len = i
                    comp_start = _get_prefix_comp(processing_times, current_seq, prefix_len)
                    jobs_suffix = current_seq[i+1:j+1] + [current_seq[i]] + current_seq[j+1:]
                else:
                    # i > j
                    # Move seq[i] to j.
                    # Prefix: seq[0:j]
                    # Job: seq[i]
                    # Middle: seq[j:i]
                    # Suffix: seq[i+1:]
                    prefix_len = j
                    comp_start = _get_prefix_comp(processing_times, current_seq, prefix_len)
                    jobs_suffix = [current_seq[i]] + current_seq[j:i] + current_seq[i+1:]
                
                ms, _ = _calc_makespan_suffix(processing_times, jobs_suffix, comp_start)
                
                if ms < current_makespan:
                    # Apply move
                    job_to_move = current_seq.pop(i)
                    if i < j:
                        insert_idx = j - 1
                    else:
                        insert_idx = j
                    current_seq.insert(insert_idx, job_to_move)
                    current_makespan = ms
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try 2-job moves (Or-opt)
            for i in range(n_jobs - 1):
                if time.time() >= deadline:
                    break
                
                for j in range(n_jobs):
                    if j == i or j == i + 1:
                        continue
                    if time.time() >= deadline:
                        break
                    
                    # Move seq[i], seq[i+1] to j
                    # This is more complex to evaluate incrementally without building the list
                    # Given time constraints, we might skip 2-job moves in the fast loop
                    # or just do a simple check.
                    # For now, let's stick to 1-job moves for speed, as they are the most common.
                    # If we have time, we can add 2-job moves in a separate pass.
                    pass
            
            # If no 1-job improvement, do a final pass with 2-job moves if time permits
            if time.time() < deadline - 1.0:
                for i in range(n_jobs - 1):
                    if time.time() >= deadline:
                        break
                    for j in range(n_jobs):
                        if j == i or j == i + 1:
                            continue
                        if time.time() >= deadline:
                            break
                        
                        # Construct new sequence
                        new_seq = current_seq[:]
                        jobs_to_move = new_seq.pop(i)
                        job2_to_move = new_seq.pop(i)
                        
                        # Determine insert index
                        # If i < j, the indices shift.
                        # Original: ... A B ... C ...
                        # Move A B to after C (index j in original, but j shifts if i < j)
                        # This is tricky. Let's just build the new seq and calculate full makespan for 2-moves
                        # to avoid bugs, as 2-moves are less frequent.
                        
                        if i < j:
                            # i, i+1 removed. j was after i+1.
                            # New index for j is j - 2
                            insert_idx = j - 2
                        else:
                            # i, i+1 removed. j was before i.
                            # New index for j is j
                            insert_idx = j
                        
                        new_seq.insert(insert_idx, jobs_to_move)
                        new_seq.insert(insert_idx + 1, job2_to_move)
                        
                        ms = _calc_makespan(processing_times, new_seq)
                        if ms < current_makespan:
                            current_seq = new_seq
                            current_makespan = ms
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
