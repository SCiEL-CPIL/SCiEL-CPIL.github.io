"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_makespan(perm, pt):
    """
    Compute the makespan for a given permutation.
    Uses a vectorized-like approach with a loop over jobs but vectorized over machines 
    where possible, though for flow shop the dependency is sequential per machine.
    We use the standard dynamic programming recurrence:
    C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[perm[j], m]
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # C[j, m] is completion time of job j (0-indexed in perm) on machine m
    # We can compute this row by row or using a 2D array
    C = np.zeros((n_jobs, n_machines))
    
    for j in range(n_jobs):
        job_id = perm[j]
        for m in range(n_machines):
            if j == 0:
                if m == 0:
                    C[j, m] = pt[job_id, m]
                else:
                    C[j, m] = C[j, m-1] + pt[job_id, m]
            else:
                if m == 0:
                    C[j, m] = C[j-1, m] + pt[job_id, m]
                else:
                    C[j, m] = np.maximum(C[j-1, m], C[j, m-1]) + pt[job_id, m]
                    
    return C[-1, -1]

def _tabu_search(perm, pt, time_limit):
    """
    Tabu Search with Insertion moves.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    current_perm = np.array(perm)
    current_mk = _compute_makespan(current_perm, pt)
    best_perm = current_perm.copy()
    best_mk = current_mk
    
    tabu_list = {}
    tabu_tenure = max(10, n_jobs // 2)
    
    start_time = time.time()
    iteration = 0
    
    # Precompute some stats for move selection if needed, but random is fine for TS
    while time.time() - start_time < time_limit:
        # Generate a batch of candidate moves to evaluate in parallel?
        # For TS, we usually pick the best non-tabu move.
        # To speed up, we can sample a subset of moves or use a heuristic.
        # Let's sample K random moves and pick the best.
        
        K = 50
        best_delta = float('inf')
        best_move = None
        
        for _ in range(K):
            i = np.random.randint(n_jobs)
            j = np.random.randint(n_jobs)
            if i == j or i == j + 1:
                continue
            
            # Move job from i to j
            # If i < j, remove i, insert at j (index j in new list is j because i removed)
            # If i > j, remove i, insert at j (index j in new list is j)
            
            new_perm = np.delete(current_perm, i)
            if i < j:
                new_perm = np.insert(new_perm, j, current_perm[i])
            else:
                new_perm = np.insert(new_perm, j, current_perm[i])
                
            # Check tabu
            # Tabu key: (job_id, new_position) or (job_id, old_position)
            # Let's use (job_id, new_position)
            key = (current_perm[i], j)
            
            if key in tabu_list:
                if tabu_list[key] > iteration - tabu_tenure:
                    continue
            
            # Evaluate
            new_mk = _compute_makespan(new_perm, pt)
            delta = new_mk - current_mk
            
            # Aspiration criterion: accept if it improves best_mk
            if delta < best_delta or new_mk < best_mk:
                best_delta = delta
                best_move = (i, j, new_perm, new_mk)
                
        if best_move is None:
            # All sampled moves were tabu or invalid, try a random move ignoring tabu
            i = np.random.randint(n_jobs)
            j = np.random.randint(n_jobs)
            if i == j or i == j + 1:
                continue
            new_perm = np.delete(current_perm, i)
            if i < j:
                new_perm = np.insert(new_perm, j, current_perm[i])
            else:
                new_perm = np.insert(new_perm, j, current_perm[i])
            new_mk = _compute_makespan(new_perm, pt)
            best_move = (i, j, new_perm, new_mk)
            best_delta = new_mk - current_mk
            
        i, j, new_perm, new_mk = best_move
        current_perm = new_perm
        current_mk = new_mk
        
        # Update tabu list
        # The move that took us here is now tabu for the reverse direction
        # Reverse move: move job current_perm[j] (which is the moved job) back to i
        # But positions have shifted. 
        # It's simpler to mark the move (job_id, pos) as tabu.
        job_id = current_perm[j]
        tabu_list[job_id] = iteration + tabu_tenure
        
        if current_mk < best_mk:
            best_mk = current_mk
            best_perm = current_perm.copy()
            
        iteration += 1
        
    return best_perm, best_mk

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Tabu Search.
    """
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s to ensure return
    deadline = start_time + 145.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initial solution: NEH Heuristic
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    init_order = list(np.argsort(-totals))
    
    # Refine NEH with local search on the initial solution? 
    # Or just run TS from multiple starts?
    # TS is generally robust. Let's do 2-3 restarts with different initializations.
    
    best_perm = init_order
    best_mk = _compute_makespan(best_perm, processing_times)
    
    # Time per restart
    n_restarts = 3
    time_per_restart = (deadline - time.time()) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate diverse initial solution
        if r == 0:
            current_init = init_order
        elif r == 1:
            # SPT
            current_init = list(np.argsort(totals))
        else:
            # Random
            current_init = list(range(n_jobs))
            random.shuffle(current_init)
            
        # Run TS
        ts_perm, ts_mk = _tabu_search(current_init, processing_times, time_per_restart)
        
        if ts_mk < best_mk:
            best_mk = ts_mk
            best_perm = ts_perm
            
    return list(best_perm)
# EVOLVE-BLOCK-END
