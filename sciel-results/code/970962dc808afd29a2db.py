"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _makespan(perm, processing_times):
    """
    Efficiently calculate makespan for a given permutation.
    perm: list of job indices
    processing_times: np.ndarray [n_jobs, n_machines]
    """
    n_machines = processing_times.shape[1]
    last_machine_time = [0] * n_machines
    max_completion = 0
    
    for job in perm:
        current_job_time = 0
        for m in range(n_machines):
            start_time = max(current_job_time, last_machine_time[m])
            end_time = start_time + processing_times[job, m]
            current_job_time = end_time
            last_machine_time[m] = end_time
            if end_time > max_completion:
                max_completion = end_time
    return max_completion

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Deterministic random seed for reproducibility
    random.seed(42)
    
    # 1. Initial Solution: SPT (Shortest Processing Time)
    # Sum processing times for each job
    totals = processing_times.sum(axis=1)
    initial_perm = list(np.argsort(totals))
    
    current_perm = initial_perm
    current_ms = _makespan(current_perm, processing_times)
    
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    T = max(1.0, current_ms / 5.0)
    cooling_rate = 0.999
    min_T = 1e-5
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Generate neighbor: Insertion move
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        job_to_move = current_perm[i]
        new_perm = current_perm[:i] + current_perm[i+1:]
        
        if i < j:
            j -= 1
            
        new_perm = new_perm[:j] + [job_to_move] + new_perm[j:]
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        # Cool down
        T *= cooling_rate
        
    return best_perm
# EVOLVE-BLOCK-END
