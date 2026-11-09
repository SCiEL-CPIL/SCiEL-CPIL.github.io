"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = (completion[m] if m > 0 else 0) + processing_times[job, m]
            if m > 0:
                # In flow shop, job must wait for previous machine to finish
                # Actually, the standard flow shop constraint is:
                # C[job][m] = max(C[job][m-1], C[prev_job][m]) + P[job][m]
                # Let's use a proper 2D array
                pass
    # Recompute properly
    C = np.zeros((n_jobs, n_machines))
    for i, job in enumerate(perm):
        for m in range(n_machines):
            if m == 0:
                C[i, m] = C[i-1, m] + processing_times[job, m] if i > 0 else processing_times[job, m]
            else:
                C[i, m] = max(C[i, m-1], C[i-1, m] if i > 0 else 0) + processing_times[job, m]
    return C[-1, -1]


def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # NEH heuristic: sort by decreasing total processing time
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insertion heuristic: try inserting each job at best position
    # Start with first job
    current_perm = [order[0]]
    
    for job in order[1:]:
        best_pos = 0
        best_makespan = np.inf
        for pos in range(len(current_perm) + 1):
            candidate = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    
    # Local search: pairwise swaps (2-opt style)
    improved = True
    while improved:
        improved = False
        n = len(current_perm)
        for i in range(n):
            for j in range(i + 1, n):
                # Try swapping i and j
                candidate = current_perm[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _makespan(candidate, processing_times) < _makespan(current_perm, processing_times):
                    current_perm = candidate
                    improved = True
    
    return current_perm
# EVOLVE-BLOCK-END
