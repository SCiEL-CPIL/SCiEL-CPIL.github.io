"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation of job indices."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = completion[m] + processing_times[job, m]
    return completion[-1]


def _neh_heuristic(processing_times, order):
    """NEH constructive heuristic given an initial job order."""
    n_jobs = processing_times.shape[0]
    schedule = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(schedule) + 1):
            candidate = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        schedule = schedule[:best_pos] + [job] + schedule[best_pos:]
    return schedule


def _local_search_2opt(perm, processing_times, max_iter=1000):
    """Local search with 2-opt moves (swap two jobs)."""
    n = len(perm)
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        for i in range(n):
            for j in range(i + 1, n):
                candidate = current_perm[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, processing_times)
                if ms < current_ms:
                    current_perm = candidate
                    current_ms = ms
                    improved = True
    return current_perm, current_ms


def order_jobs(processing_times):
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    
    # Initial ordering by total processing time (descending)
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    # Apply NEH heuristic
    schedule = _neh_heuristic(processing_times, order)
    
    # Local search with 2-opt
    best_perm, best_ms = _local_search_2opt(schedule, processing_times, max_iter=500)
    
    return best_perm
# EVOLVE-BLOCK-END
