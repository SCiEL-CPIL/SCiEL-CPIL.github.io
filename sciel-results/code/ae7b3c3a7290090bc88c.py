"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    comp = [0.0] * n_machines
    for j in perm:
        row = pt[j]
        for m in range(n_machines):
            if m == 0:
                comp[m] += row[m]
            else:
                comp[m] = max(comp[m], comp[m-1]) + row[m]
    return comp[-1]

def _makespan_suffix(perm, pt, start, comp_prev):
    """Compute makespan from index 'start' given completion times 'comp_prev'."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if start >= n_jobs:
        return comp_prev[-1] if n_machines > 0 else 0.0
    
    comp = comp_prev[:]
    for k in range(start, n_jobs):
        j = perm[k]
        row = pt[j]
        for m in range(n_machines):
            if m == 0:
                comp[m] += row[m]
            else:
                comp[m] = max(comp[m], comp[m-1]) + row[m]
    return comp[-1]

def _compute_prefix(perm, pt, end_idx):
    """Compute completion times up to index 'end_idx' (exclusive)."""
    n_machines = pt.shape[1]
    comp = [0.0] * n_machines
    for k in range(end_idx):
        j = perm[k]
        row = pt[j]
        for m in range(n_machines):
            if m == 0:
                comp[m] += row[m]
            else:
                comp[m] = max(comp[m], comp[m-1]) + row[m]
    return comp

def _neh_order(processing_times):
    """NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm

def _sa_search(perm, pt, time_limit, seed=0):
    """Simulated Annealing with adjacent swaps."""
    rng = np.random.default_rng(seed)
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
        
    current_perm = list(perm)
    current_ms = _makespan(current_perm, pt)
    
    # Initial temperature
    # Scale with total processing time
    total_pt = pt.sum()
    if total_pt == 0:
        return current_perm
    T = total_pt / 5.0
    T_min = 0.01
    alpha = 0.9999
    
    start_time = time.time()
    best_perm = list(current_perm)
    best_ms = current_ms
    
    n_machines = pt.shape[1]
    
    while time.time() - start_time < time_limit:
        i = rng.integers(0, n_jobs - 1)
        
        # Swap
        current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
        
        # Compute prefix up to i
        comp_prev = _compute_prefix(current_perm, pt, i)
        
        # Compute suffix from i
        new_ms = _makespan_suffix(current_perm, pt, i, comp_prev)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = list(current_perm)
        else:
            # Revert
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally reheat if stuck, but for simplicity, keep going
            # If we want to escape deep local minima, we could restart SA
            # But for now, let's just continue with T_min
            
    return best_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + SA.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate initial solutions
    # 1. NEH
    perm_neh = _neh_order(processing_times)
    ms_neh = _makespan(perm_neh, processing_times)
    
    # 2. Random
    rng = np.random.default_rng(42)
    perm_rand = list(rng.permutation(n_jobs))
    ms_rand = _makespan(perm_rand, processing_times)
    
    # 3. NEH with perturbation (swap first two if better?)
    # Just use NEH and Random as bases for SA
    
    best_perm = perm_neh if ms_neh < ms_rand else perm_rand
    best_ms = min(ms_neh, ms_rand)
    
    # Run SA from multiple starts
    # Allocate time
    elapsed = time.time() - start_time
    remaining_time = deadline - start_time - elapsed
    if remaining_time <= 0:
        return best_perm
        
    # 3 SA runs
    n_runs = 3
    time_per_run = remaining_time / n_runs
    
    for i in range(n_runs):
        if time.time() > deadline:
            break
            
        # Choose starting point
        if i == 0:
            start_perm = perm_neh
        elif i == 1:
            start_perm = perm_rand
        else:
            # Perturbed NEH
            start_perm = list(perm_neh)
            # Randomly reverse a segment
            if n_jobs > 1:
                idx1 = rng.integers(0, n_jobs - 1)
                idx2 = rng.integers(idx1 + 1, n_jobs)
                start_perm[idx1:idx2+1] = start_perm[idx1:idx2+1][::-1]
        
        # Run SA
        # Leave a small buffer for the next run
        actual_time_limit = time_per_run - 0.1 
        if actual_time_limit < 1.0:
            actual_time_limit = 1.0
            
        sa_result = _sa_search(start_perm, processing_times, actual_time_limit, seed=i)
        sa_ms = _makespan(sa_result, processing_times)
        
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_perm = sa_result
            
    return best_perm
# EVOLVE-BLOCK-END
