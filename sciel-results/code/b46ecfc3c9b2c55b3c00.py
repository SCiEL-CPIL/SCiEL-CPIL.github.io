"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for j in perm:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m-1] if m > 0 else 0) + processing_times[j, m]
    return completion[-1]


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # NEH heuristic: order jobs by total processing time descending
    # For each job in order, insert it at the best position in the current sequence
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _calc_makespan(processing_times, new_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    best_perm = perm[:]
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # Local search: 2-opt style swaps
    while time.time() < deadline:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                new_perm = best_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _calc_makespan(processing_times, new_perm)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
        if not improved:
            break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
