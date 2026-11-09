"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, perm):
    """Compute makespan for a given permutation of jobs."""
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    # cum[m] = completion time on machine m for the last processed job
    cum = np.zeros(processing_times.shape[1])
    for j in perm:
        for m in range(processing_times.shape[1]):
            cum[m] = max(cum[m], cum[m-1] if m > 0 else 0) + processing_times[j, m]
    return cum[-1]

def _neh_initial(processing_times):
    """NEH heuristic: build permutation by inserting jobs in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for j in order:
        # Find best position to insert j
        best_pos = 0
        best_ms = float('inf')
        current_len = len(perm)
        for pos in range(current_len + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [j] + perm[best_pos:]
    return perm

def _neighborhood_insertion(perm):
    """Generate a neighbor by removing one job and reinserting it at another position."""
    n = len(perm)
    if n <= 1:
        return perm
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    if i == j:
        return perm
    job = perm[i]
    new_perm = perm[:i] + perm[i+1:]
    if j > i:
        j -= 1
    j = min(j, len(new_perm))
    new_perm = new_perm[:j] + [job] + new_perm[j:]
    return new_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH heuristic + local search.
    Returns a permutation of job indices.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initial solution using NEH heuristic
    best_perm = _neh_initial(processing_times)
    best_ms = _makespan(processing_times, best_perm)
    
    # Local search with simulated annealing
    current_perm = best_perm
    current_ms = best_ms
    
    # Temperature parameters
    T0 = best_ms * 0.01 + 1.0
    T_min = 1e-6
    cooling_rate = 0.999
    
    T = T0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Generate neighbor
        new_perm = _neighborhood_insertion(current_perm)
        new_ms = _makespan(processing_times, new_perm)
        
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept improvement
            current_perm = new_perm
            current_ms = new_ms
            no_improve_count = 0
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm
        else:
            # Accept with probability exp(-delta / T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T *= cooling_rate
        
        # If too many iterations without improvement, restart with perturbation
        if no_improve_count > max_no_improve:
            # Perturb the best solution
            current_perm = list(best_perm)
            for _ in range(min(3, n_jobs // 2)):
                current_perm = _neighborhood_insertion(current_perm)
            current_ms = _makespan(processing_times, current_perm)
            T = T0
            no_improve_count = 0
        
        # Check time limit
        if time.time() >= deadline:
            break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
