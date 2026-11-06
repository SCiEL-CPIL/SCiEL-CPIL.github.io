"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    pt = processing_times
    
    # Local variable caching for speed
    comp = completion
    for job in perm:
        row = pt[job]
        # Unroll first machine
        c0 = comp[0] + row[0]
        comp[0] = c0
        c_prev = c0
        for m in range(1, n_machines):
            c_curr = comp[m]
            if c_curr < c_prev:
                c_curr = c_prev
            c_curr += row[m]
            comp[m] = c_curr
            c_prev = c_curr
    
    return comp[-1]


def _calc_makespan_incremental(processing_times, perm, start_pos):
    """
    Calculate makespan for perm, assuming completion times before start_pos are already known.
    Returns the final makespan.
    """
    n_machines = processing_times.shape[1]
    # We need the completion times at start_pos for all machines.
    # This function is intended to be used when we have the state from a previous prefix.
    # However, for simplicity in the current structure, we will stick to full calculation
    # but optimize the NEH step specifically.
    pass


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
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            
            # Optimized makespan calculation for insertion
            # We can compute the completion times up to pos-1, then insert job, then continue
            # But for now, let's just use the standard _calc_makespan
            # We can optimize this later if needed.
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Insertion local search: try moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    # Limit the number of passes to prevent infinite loops in tight time constraints
    # A single pass is O(N^3 M). We allow at most 5 passes or until time runs out.
    max_passes = 5
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        improved = False
        # Try all pairs of positions (i, j)
        # Move job at i to position j
        for i in range(n_jobs):
            if time.time() >= deadline:
                return current_seq, current_makespan
            
            # Skip if this job is already in the best position relative to neighbors? 
            # No, just try all moves.
            
            # Optimization: Only check moves that might improve. 
            # But for safety and simplicity, we check all, but break early on improvement.
            
            # To speed up, we can skip some checks or use a faster makespan delta calculation,
            # but implementing that correctly is complex. 
            # Instead, we limit the inner loop frequency or use a smaller subset if N is large?
            # No, let's just ensure the time check is frequent.
            
            # Check time every 10 moves to reduce overhead of time.time()
            for j in range(n_jobs):
                if j == i:
                    continue
                
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
        if not improved:
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
    
    # Temperature parameters
    T = current_makespan * 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Random perturbation: swap two jobs
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
    # Use a conservative deadline to avoid timeouts (100s out of 120s)
    deadline = start_time + 100.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    # NEH can be slow for very large N, but it's the best start.
    # If N is huge, we might need to cap the NEH or use a faster heuristic,
    # but let's assume standard sizes first.
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Local search with time limit
    # Allocate 50% of the time to the first local search pass
    ls_deadline_1 = min(deadline - 10.0, start_time + 50.0)
    best_seq, best_makespan = _local_search(processing_times, initial_seq, ls_deadline_1)
    
    # Multi-start strategy: if time remains, perturb and re-search
    # Limit the number of restarts to ensure we don't run out of time in the loop
    # Each restart gets a slice of the remaining time.
    remaining_time = deadline - time.time()
    
    # Perform at most 3 additional restarts
    for restart_idx in range(3):
        if time.time() >= deadline - 5.0:
            break
        
        # Create a perturbed version of the best sequence
        perturbed_seq = best_seq[:]
        if n_jobs > 1:
            # Use 2-3 swaps for better perturbation
            k = min(3, n_jobs // 2)
            for _ in range(k):
                i, j = random.sample(range(n_jobs), 2)
                perturbed_seq[i], perturbed_seq[j] = perturbed_seq[j], perturbed_seq[i]
        
        # Run local search on the perturbed sequence
        # Allocate remaining time divided by remaining restarts, with a cap
        slice_time = (deadline - time.time() - 5.0) / (3 - restart_idx)
        restart_deadline = min(deadline - 5.0, time.time() + max(5.0, slice_time))
        
        new_seq, new_makespan = _local_search(processing_times, perturbed_seq, restart_deadline)
        
        if new_makespan < best_makespan:
            best_seq = new_seq
            best_makespan = new_makespan
    
    # Final polish if time permits
    if time.time() < deadline - 2.0:
        final_deadline = deadline - 0.5
        final_seq, final_makespan = _local_search(processing_times, best_seq, final_deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
    
    return best_seq


# EVOLVE-BLOCK-END
