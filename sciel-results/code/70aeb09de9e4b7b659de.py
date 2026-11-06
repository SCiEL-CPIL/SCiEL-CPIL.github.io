"""Flow-shop makespan minimization via simulated annealing with incremental makespan updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_makespan(pt, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in perm:
        for m in range(n_machines):
            completion[m] = completion[m] + pt[job, m]
    return completion[-1]


def _incremental_makespan_after_swap(pt, perm, i, j):
    """
    Compute the change in makespan when swapping positions i and j in perm (i < j).
    Returns (new_makespan, old_makespan) efficiently by only recomputing from min(i,j) onwards.
    """
    n_jobs, n_machines = pt.shape
    old_perm = list(perm)
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    
    # Find the first position where the sequence differs
    start = min(i, j)
    
    # Compute completion times from start onwards for both permutations
    # We need the state (completion times on all machines) at position 'start'
    
    # Compute completion state at position 'start' for the original permutation
    comp_old = np.zeros(n_machines, dtype=np.float64)
    for pos in range(start):
        job = old_perm[pos]
        for m in range(n_machines):
            comp_old[m] += pt[job, m]
    
    # Now simulate from 'start' onwards for both old and new permutations
    # Old permutation
    comp_o = comp_old.copy()
    for pos in range(start, n_jobs):
        job = old_perm[pos]
        for m in range(n_machines):
            comp_o[m] += pt[job, m]
    old_ms = comp_o[-1]
    
    # New permutation
    comp_n = comp_old.copy()
    for pos in range(start, n_jobs):
        job = new_perm[pos]
        for m in range(n_machines):
            comp_n[m] += pt[job, m]
    new_ms = comp_n[-1]
    
    return new_ms, old_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using simulated annealing
    with a fast incremental makespan update mechanism for adjacent swaps.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # --- Initial solution: NEH heuristic ---
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    # Build sequence using NEH rule
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            ms = _compute_makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    current_perm = list(seq)
    current_ms = _compute_makespan(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # --- Simulated Annealing ---
    T = max(1.0, current_ms * 0.01)  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    n_iterations = 0
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Random adjacent swap
        if n_jobs <= 2:
            i = 0
            j = 1
        else:
            i = np.random.randint(0, n_jobs - 1)
            j = i + 1
        
        new_ms, old_ms = _incremental_makespan_after_swap(processing_times, current_perm, i, j)
        delta = new_ms - old_ms
        
        if delta < 0:
            # Accept improvement
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
        else:
            # Metropolis acceptance criterion
            if np.random.random() < np.exp(-delta / T):
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_ms = new_ms
        
        T *= cooling_rate
        n_iterations += 1
    
    return best_perm
# EVOLVE-BLOCK-END
