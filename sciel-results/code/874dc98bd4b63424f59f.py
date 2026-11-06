"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return 0.0
    # Compute completion times
    # C[j][m] = completion time of job j on machine m
    # We can compute this incrementally
    # For a given permutation, job at position p is processed on machine m after
    # max(C_prev_job_on_m, C_current_job_on_prev_m)
    
    # Use a 2D array: rows = positions in permutation, cols = machines
    # Actually, let's just track the last completion time on each machine
    # and for each job in the permutation, compute its completion on each machine
    
    last_mach_time = np.zeros(n_machines)
    for job in perm:
        last_job_time = 0.0
        for m in range(n_machines):
            start = max(last_mach_time[m], last_job_time)
            last_job_time = start + pt[job, m]
            last_mach_time[m] = last_job_time
    return last_mach_time[-1]


def _neh_initial(pt):
    """NEH heuristic for initial construction."""
    n_jobs, n_machines = pt.shape
    total_times = pt.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(trial, pt)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    return seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction
    2. Local search with adjacent swap and insertion moves
    3. Deadline guard to ensure we return within time limits
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    current_time = time.time()
    
    # Stage 1: NEH initial solution
    perm = _neh_initial(processing_times)
    best_perm = list(perm)
    best_ms = _makespan(perm, processing_times)
    
    # Stage 2: Local search with insertion moves
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n_jobs):
                if time.time() > deadline - 5:
                    break
                # Try swapping jobs at positions i and j
                trial = list(perm)
                trial[i], trial[j] = trial[j], trial[i]
                ms = _makespan(trial, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    perm = trial
                    improved = True
        
        # Also try insertion moves
        for i in range(n_jobs):
            if time.time() > deadline - 5:
                break
            job = perm[i]
            for j in range(n_jobs):
                if j == i or time.time() > deadline - 5:
                    continue
                trial = list(perm)
                del trial[i]
                # Insert at position j (adjust if j > i)
                insert_pos = j if j < i else j - 1
                trial.insert(insert_pos, job)
                ms = _makespan(trial, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    perm = trial
                    improved = True
    
    # Stage 3: Final polish with adjacent swaps only
    improved = True
    while improved and (time.time() < deadline - 2):
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline - 2:
                break
            trial = list(perm)
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                perm = trial
                improved = True
    
    best_perm = list(perm)
    return best_perm
# EVOLVE-BLOCK-END
