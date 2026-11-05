"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given permutation of job indices."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    # completion[j][m] = completion time of job j on machine m
    # We track the completion time of each machine for the current job sequence
    machine_time = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            machine_time[m] = max(machine_time[m], machine_time[m-1] if m > 0 else 0) + processing_times[job, m]
    return machine_time[-1]


def _makespan_fast(processing_times, order):
    """Faster makespan computation using vectorized operations where possible."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Pre-extract the submatrix for the ordered jobs
    pt = processing_times[order]
    
    # For each machine, the completion time is the sum of processing times
    # plus the maximum of previous machines' completion times at each step.
    # We can compute this iteratively but vectorize across jobs where possible.
    
    # Simpler approach: for small n, just loop
    machine_time = np.zeros(n_machines)
    for j in range(n_jobs):
        for m in range(n_machines):
            prev = machine_time[m-1] if m > 0 else 0.0
            machine_time[m] = max(machine_time[m], prev) + pt[j, m]
    return machine_time[-1]


def _neh(processing_times):
    """NEH heuristic: order by decreasing total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence incrementally
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _palmer(processing_times):
    """Palmer's priority rule: sort by decreasing (n+1-j)*t_j1 - j*t_jn."""
    n_jobs, n_machines = processing_times.shape
    if n_machines == 0:
        return list(range(n_jobs))
    priorities = np.zeros(n_jobs)
    for j in range(n_jobs):
        priorities[j] = (n_jobs + 1 - j) * processing_times[j, 0] - j * processing_times[j, n_machines - 1]
    return list(np.argsort(-priorities))


def _lpt(processing_times):
    """Longest Processing Time first."""
    return list(np.argsort(-processing_times.sum(axis=1)))


def _spt(processing_times):
    """Shortest Processing Time first."""
    return list(np.argsort(processing_times.sum(axis=1)))


def _local_search(processing_times, order, time_limit=10.0):
    """Adjacent swap and segment reversal local search with time limit."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            trial = best_order[:]
            trial[i], trial[i+1] = trial[i+1], trial[i]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_order = trial
                best_ms = ms
                improved = True
        
        # Try segment reversals (length 2 and 3)
        for seg_len in [2, 3]:
            for i in range(n_jobs - seg_len + 1):
                if (time.time() - start_time) >= time_limit:
                    break
                trial = best_order[:]
                trial[i:i+seg_len] = reversed(trial[i:i+seg_len])
                ms = _makespan(processing_times, trial)
                if ms < best_ms:
                    best_order = trial
                    best_ms = ms
                    improved = True
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multi-start heuristic (NEH, Palmer, LPT, SPT) followed by local search.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate initial solutions from multiple heuristics
    initial_solutions = []
    
    # NEH
    initial_solutions.append(_neh(processing_times))
    
    # Palmer
    initial_solutions.append(_palmer(processing_times))
    
    # LPT
    initial_solutions.append(_lpt(processing_times))
    
    # SPT
    initial_solutions.append(_spt(processing_times))
    
    # Add a few random permutations
    rng = random.Random(42)
    for _ in range(3):
        perm = list(range(n_jobs))
        rng.shuffle(perm)
        initial_solutions.append(perm)
    
    # Evaluate all initial solutions
    best_order = None
    best_ms = float('inf')
    for sol in initial_solutions:
        ms = _makespan(processing_times, sol)
        if ms < best_ms:
            best_ms = ms
            best_order = sol
    
    # Run local search on the best initial solution
    if (time.time() - start_time) < deadline - 5.0:
        time_remaining = deadline - 5.0 - (time.time() - start_time)
        time_remaining = max(time_remaining, 1.0)
        best_order = _local_search(processing_times, best_order, time_limit=time_remaining)
        best_ms = _makespan(processing_times, best_order)
    
    # If we still have time, try local search on other initial solutions
    if (time.time() - start_time) < deadline - 5.0:
        for sol in initial_solutions:
            if sol == best_order:
                continue
            if (time.time() - start_time) >= deadline - 5.0:
                break
            time_remaining = deadline - 5.0 - (time.time() - start_time)
            if time_remaining < 1.0:
                break
            candidate = _local_search(processing_times, sol, time_limit=time_remaining)
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_order = candidate
    
    return list(best_order)
# EVOLVE-BLOCK-END
