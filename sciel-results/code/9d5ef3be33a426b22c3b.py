"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # completion[m] = time when machine m becomes free
    completion = np.zeros(n_machines)
    for job_idx in perm:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job_idx, m]
            else:
                completion[m] = max(completion[m], completion[m - 1]) + processing_times[job_idx, m]
    return completion[-1]


def _makespan_fast(perm, processing_times):
    """Fast makespan computation for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # Use Python lists for faster access in tight loops
    pt = processing_times
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = pt[job_idx]
        for m in range(n_machines):
            if m == 0:
                completion[m] += row[m]
            else:
                prev = completion[m - 1]
                cur = completion[m]
                if prev > cur:
                    completion[m] = prev + row[m]
                else:
                    completion[m] = cur + row[m]
    return completion[-1]


def _neh_order(processing_times):
    """NEH heuristic: insert jobs one by one in order of decreasing total processing time."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _adjacent_swap_neighborhood(perm, processing_times, n_iterations=2000, deadline=None):
    """Local search using adjacent swaps."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, processing_times)
    
    improved = True
    while improved and n_iterations > 0:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_jobs - 1):
            # Try swapping i and i+1
            new_perm = current_perm.copy()
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
            new_ms = _makespan_fast(new_perm, processing_times)
            if new_ms < current_ms - 1e-12:
                current_perm = new_perm
                current_ms = new_ms
                improved = True
                n_iterations -= 1
                if n_iterations <= 0:
                    break
                # Restart scan from the beginning since the permutation changed
                break
    
    return current_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + local search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 140s out of 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH constructive heuristic
    perm = _neh_order(processing_times)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, processing_times)
    
    # Step 2: Local search with adjacent swaps
    perm = _adjacent_swap_neighborhood(perm, processing_times, n_iterations=10000, deadline=deadline)
    ms = _makespan_fast(perm, processing_times)
    if ms < best_ms:
        best_perm = list(perm)
        best_ms = ms
    
    # Step 3: If time remains, do more local search passes
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try more rounds of local search from perturbed starts
        rng = np.random.default_rng(42)
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Perturb the best permutation
            perturbed = list(best_perm)
            # Randomly reverse a segment
            i = rng.integers(0, n_jobs - 1)
            j = rng.integers(i + 1, n_jobs)
            perturbed[i:j + 1] = perturbed[i:j + 1][::-1]
            
            # Local search from perturbed start
            improved_perm = _adjacent_swap_neighborhood(perturbed, processing_times, n_iterations=2000, deadline=deadline)
            improved_ms = _makespan_fast(improved_perm, processing_times)
            if improved_ms < best_ms:
                best_perm = list(improved_perm)
                best_ms = improved_ms
    
    return best_perm
# EVOLVE-BLOCK-END
