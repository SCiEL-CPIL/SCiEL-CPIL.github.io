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

def _local_search(perm, processing_times, time_limit=None, start_time=None):
    """
    Perform Local Search with First Improvement strategy using Swap and Insertion moves.
    """
    n_jobs = len(perm)
    if start_time is None:
        start_time = time.time()
    
    current_ms = _makespan(perm, processing_times)
    improved = True
    
    while improved:
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        improved = False
        # 1. Swap Moves
        for i in range(n_jobs - 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            for j in range(i + 1, n_jobs):
                new_perm = perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan(new_perm, processing_times)
                if new_ms < current_ms:
                    perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break # First improvement, restart scan
            if improved:
                break
        
        if not improved:
            # 2. Insertion Moves
            for i in range(n_jobs):
                if time_limit and (time.time() - start_time) > time_limit:
                    break
                job = perm[i]
                # Remove job
                remaining = perm[:i] + perm[i+1:]
                # Try inserting at all positions
                for j in range(len(remaining) + 1):
                    new_perm = remaining[:j] + [job] + remaining[j:]
                    new_ms = _makespan(new_perm, processing_times)
                    if new_ms < current_ms:
                        perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break # First improvement, restart scan
                if improved:
                    break
                    
    return perm, current_ms

def _perturb(perm, processing_times, intensity=3):
    """
    Perturb the permutation by performing 'intensity' random swaps.
    """
    n_jobs = len(perm)
    perm = perm[:]
    for _ in range(intensity):
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i != j:
            perm[i], perm[j] = perm[j], perm[i]
    return perm

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # Deterministic random seed for reproducibility
    random.seed(42)
    
    # 1. Initial Solutions
    totals = processing_times.sum(axis=1)
    
    # SPT
    perm_spt = list(np.argsort(totals))
    # LPT
    perm_lpt = list(np.argsort(-totals))
    # NN based on first machine
    perm_nn_first = list(np.argsort(processing_times[:, 0]))
    # NN based on last machine
    perm_nn_last = list(np.argsort(processing_times[:, -1]))
    
    candidates = [perm_spt, perm_lpt, perm_nn_first, perm_nn_last]
    
    best_perm = None
    best_ms = float('inf')
    
    for p in candidates:
        ms = _makespan(p, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = p[:]
            
    # 2. Iterated Local Search
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    while (time.time() - start_time) < time_limit:
        # Perturb the best solution
        current_perm = _perturb(best_perm, processing_times, intensity=max(2, n_jobs // 10))
        
        # Local Search
        local_perm, local_ms = _local_search(current_perm, processing_times, time_limit, start_time)
        
        if local_ms < best_ms:
            best_ms = local_ms
            best_perm = local_perm[:]
            
        # Optional: If no improvement in a while, increase perturbation?
        # For now, keep constant.
        
    return best_perm
# EVOLVE-BLOCK-END
