"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """
    Calculate makespan for a given order of jobs.
    processing_times: np.ndarray of shape (n_jobs, n_machines)
    order: list of job indices
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Fast calculation using vectorized numpy operations where possible
    # However, flow shop completion times are sequential per machine.
    # C[j][m] = C[j][m-1] + max(C[j-1][m], C[j][m-1] + P[j][m])? No.
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    
    # We can compute this in a loop over jobs, but vectorize over machines if n_machines is large?
    # Actually, for each job j in order:
    #   start_time_m0 = 0
    #   for m in range(n_machines):
    #       start_time = max(prev_machine_finish_for_job, prev_job_finish_for_machine)
    #       finish_time = start_time + P[j][m]
    
    # Let's use a simple loop which is often fast enough for n<100.
    # To be safe and fast, we'll stick to a tight Python loop or try to vectorize.
    # Given the 120s limit, and typical n_jobs (e.g., 20-100), a Python loop might be slow if we do millions of evals.
    # Let's try a slightly optimized Python loop.
    
    if n_jobs == 0:
        return 0.0
        
    # C_prev_machine[j] is the finish time of job j on machine m-1
    # C_prev_job[m] is the finish time of job j-1 on machine m
    
    # We track the finish time of the *previous job* on each machine
    prev_job_finish = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in order:
        # Finish time of this job on the previous machine (initially 0 for first machine)
        prev_machine_finish = 0.0
        for m in range(n_machines):
            # The job can start on machine m after:
            # 1. The previous job finished on machine m (prev_job_finish[m])
            # 2. This job finished on the previous machine (prev_machine_finish)
            start_time = max(prev_job_finish[m], prev_machine_finish)
            finish_time = start_time + processing_times[job_idx, m]
            prev_machine_finish = finish_time
            prev_job_finish[m] = finish_time
        makespan = prev_machine_finish # The finish time on the last machine for the last job
        
    return makespan

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    # Initial solution: NEH heuristic (Longest Processing Time) is a good start
    # But we can also try a few random starts or just stick to one good heuristic start
    # to save time for the search.
    
    # 1. Generate initial solution using LPT (Longest Processing Time)
    init_order = list(np.argsort(-processing_times.sum(axis=1)))
    best_order = init_order.copy()
    best_score = _makespan(processing_times, best_order)
    
    # 2. Simulated Annealing
    # Parameters
    T = 100.0 # Initial temperature
    alpha = 0.99 # Cooling rate
    T_min = 0.01
    
    # We will use insertion moves: pick a random job, remove it, insert it at a random new position
    # This allows the search to explore the neighborhood effectively.
    
    current_order = best_order.copy()
    current_score = best_score
    
    # Pre-compute some things? Not really, makespan is sequential.
    
    while time.time() < deadline:
        # If temperature is too low, break or restart?
        # Simple SA usually just stops when T < T_min, but we have a time limit.
        # Let's just run until time runs out, adjusting T slowly.
        
        if T < T_min:
            # If we've cooled down, we could restart with a new random solution or break.
            # Restarting might find a different basin. Let's restart with a random permutation.
            current_order = list(np.random.permutation(n_jobs))
            current_score = _makespan(processing_times, current_order)
            if current_score < best_score:
                best_score = current_score
                best_order = current_order.copy()
            T = 100.0 # Reset temperature
            
        # Generate neighbor: Insertion move
        # Pick random job to move
        i = np.random.randint(n_jobs)
        job_to_move = current_order[i]
        
        # Remove it
        current_order.pop(i)
        
        # Insert at random position
        # The new index can be 0 to n_jobs-1 (since we removed one, length is n_jobs-1)
        j = np.random.randint(n_jobs)
        current_order.insert(j, job_to_move)
        
        new_score = _makespan(processing_times, current_order)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_order = current_order.copy()
        else:
            # Deterioration: accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
            # else: reject (T is 0)
            
        T *= alpha
        
    return best_order

# EVOLVE-BLOCK-END
