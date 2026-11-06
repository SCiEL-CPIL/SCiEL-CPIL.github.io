"""
Strategy: Parallel Portfolio of Diversified Local Searches
Instead of a single constructive+local search track, we run multiple distinct
heuristic pipelines in parallel (sequentially with time-slicing).
1. NEH Constructive + Insertion Local Search
2. NEH Constructive + 2-Opt Local Search
3. Random Constructive + Insertion Local Search
4. Random Constructive + 2-Opt Local Search

This diversification helps escape the local minima that trap the single-track
greedy insertion approach. The time is divided equally among the active tracks,
and the best result found across all tracks is returned.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, processing_times):
    """
    Compute makespan for a given permutation.
    perm: list of job indices
    processing_times: numpy array of shape (n_jobs, n_machines)
    """
    n_machines = processing_times.shape[1]
    if not perm:
        return 0.0
    
    # C[j, m] = completion time of job at position j on machine m
    # We only need the previous row to compute the current row
    prev_row = np.zeros(n_machines)
    
    for job_idx in perm:
        curr_row = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                # First machine: depends only on previous job on same machine
                curr_row[m] = prev_row[m] + processing_times[job_idx, m]
            else:
                # Subsequent machines: depends on previous job on same machine 
                # AND previous machine for the same job
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[job_idx, m]
        prev_row = curr_row
        
    return prev_row[-1]

def _generate_neh_order(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for initial ordering.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # For ties, use secondary criterion: max(processing on first machine, processing on last machine)
    # Actually, standard NEH sorts by total time. If tie, larger max(t_1, t_m) first.
    # Let's implement the full NEH insertion.
    
    # Initial sort by total time
    sorted_jobs = sorted(range(n_jobs), key=lambda j: (-totals[j], -max(processing_times[j, 0], processing_times[j, -1])))
    
    current_perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_cost = float('inf')
        # Try inserting job at all positions
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cost = _makespan(new_perm, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        current_perm.insert(best_pos, job)
        
    return current_perm

def _local_search_insertion(perm, processing_times, deadline):
    """
    Local search using insertion moves.
    Remove a job and re-insert it at the best position.
    """
    current_perm = list(perm)
    n = len(current_perm)
    current_cost = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return best_perm, best_cost
            job = current_perm[i]
            # Remove job
            current_perm.pop(i)
            # Try all insertion positions
            best_pos = i
            best_cost_candidate = current_cost
            # We need to evaluate insertion costs. 
            # Optimization: We can just try all positions.
            for pos in range(len(current_perm) + 1):
                new_perm = current_perm[:pos] + [job] + current_perm[pos:]
                cost = _makespan(new_perm, processing_times)
                if cost < best_cost_candidate:
                    best_cost_candidate = cost
                    best_pos = pos
            
            if best_cost_candidate < current_cost:
                # Re-insert at best position
                current_perm.insert(best_pos, job)
                current_cost = best_cost_candidate
                improved = True
                if current_cost < best_cost:
                    best_perm = list(current_perm)
                    best_cost = current_cost
            else:
                # Put it back where it was (or best position if equal? No, just restore)
                # Actually, if no improvement, we should restore the original position to maintain state
                # But we popped it. So we insert it back at i (or best_pos if it was the same cost? No, strictly better only)
                # To be safe, if no improvement, insert back at original index i
                current_perm.insert(i, job)
                
    return best_perm, best_cost

def _local_search_2opt(perm, processing_times, deadline):
    """
    Local search using 2-opt moves (reverse segment).
    """
    current_perm = list(perm)
    n = len(current_perm)
    current_cost = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                return best_perm, best_cost
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best_perm, best_cost
                # Reverse segment [i, j]
                new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
                cost = _makespan(new_perm, processing_times)
                if cost < current_cost:
                    current_perm = new_perm
                    current_cost = cost
                    improved = True
                    if current_cost < best_cost:
                        best_perm = list(current_perm)
                        best_cost = current_cost
                    # Break inner loop? Usually yes, to restart scan from beginning or continue from current state.
                    # Breaking inner loop and restarting outer loop is standard for 2-opt.
                    break
            if improved:
                # If we made a change, break outer loop to restart scan
                break
                
    return best_perm, best_cost

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    total_budget = 140.0
    n_tracks = 4
    time_per_track = total_budget / n_tracks
    
    # Ensure random seed is consistent for reproducibility if needed, 
    # but random track benefits from randomness. 
    # We don't set seed to allow different runs to explore, 
    # but for deterministic evaluation, maybe we should? 
    # The task says "local synthetic flow-shop instances", implying fixed inputs.
    # Randomness in the "Random" track will make the result non-deterministic.
    # To be safe and consistent, let's use a fixed seed for the random track 
    # or just rely on the deterministic tracks being strong.
    # Actually, using a fixed seed makes the "Random" track deterministic.
    random.seed(42)
    
    best_perm = None
    best_cost = float('inf')
    
    # Track 1: NEH + Insertion
    deadline_1 = start_time + time_per_track
    try:
        init_perm = _generate_neh_order(processing_times)
        ls_perm, ls_cost = _local_search_insertion(init_perm, processing_times, deadline_1)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_perm = ls_perm
    except Exception:
        pass

    # Track 2: NEH + 2-Opt
    deadline_2 = start_time + 2 * time_per_track
    try:
        init_perm = _generate_neh_order(processing_times)
        ls_perm, ls_cost = _local_search_2opt(init_perm, processing_times, deadline_2)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_perm = ls_perm
    except Exception:
        pass

    # Track 3: Random + Insertion
    deadline_3 = start_time + 3 * time_per_track
    try:
        n_jobs = processing_times.shape[0]
        init_perm = list(range(n_jobs))
        random.shuffle(init_perm)
        ls_perm, ls_cost = _local_search_insertion(init_perm, processing_times, deadline_3)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_perm = ls_perm
    except Exception:
        pass

    # Track 4: Random + 2-Opt
    deadline_4 = start_time + total_budget
    try:
        n_jobs = processing_times.shape[0]
        init_perm = list(range(n_jobs))
        random.shuffle(init_perm)
        ls_perm, ls_cost = _local_search_2opt(init_perm, processing_times, deadline_4)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_perm = ls_perm
    except Exception:
        pass

    # Fallback if all failed (should not happen)
    if best_perm is None:
        best_perm = list(range(processing_times.shape[0]))
        
    return list(best_perm)
# EVOLVE-BLOCK-END
