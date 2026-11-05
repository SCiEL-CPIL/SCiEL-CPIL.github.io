"""Flow-shop permutation heuristic with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given permutation of job indices."""
    n = len(order)
    if n == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    machine_time = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        for m in range(n_machines):
            prev = machine_time[m - 1] if m > 0 else 0.0
            machine_time[m] = max(machine_time[m], prev) + processing_times[job, m]
    return machine_time[-1]


def _neh(processing_times):
    """NEH heuristic: order by decreasing total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals, kind='stable'))
    
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


def _neh_improved(processing_times):
    """NEH with improved insertion: try all positions but use a smarter initial guess."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals, kind='stable'))
    
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        # Try positions from both ends inward, but check all
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
    return list(np.argsort(-priorities, kind='stable'))


def _lpt(processing_times):
    """Longest Processing Time first."""
    return list(np.argsort(-processing_times.sum(axis=1), kind='stable'))


def _spt(processing_times):
    """Shortest Processing Time first."""
    return list(np.argsort(processing_times.sum(axis=1), kind='stable'))


def _insertion_heuristic(processing_times, initial_order):
    """Take an initial order and try to improve by removing and re-inserting each job."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    seq = list(initial_order)
    # For each job (in reverse order of the initial sequence), try removing and re-inserting
    for i in range(n_jobs - 1, -1, -1):
        job = seq[i]
        # Remove job from position i
        remaining = seq[:i] + seq[i+1:]
        best_pos = i if i < len(remaining) else len(remaining)
        best_ms = float('inf')
        for pos in range(len(remaining) + 1):
            trial = remaining[:pos] + [job] + remaining[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = remaining[:best_pos] + [job] + remaining[best_pos:]
    
    return seq


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


def _two_opt(processing_times, order, time_limit=10.0):
    """2-opt style segment reversal local search."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, n_jobs):
                if (time.time() - start_time) >= time_limit:
                    break
                trial = best_order[:i] + list(reversed(best_order[i:j+1])) + best_order[j+1:]
                ms = _makespan(processing_times, trial)
                if ms < best_ms:
                    best_order = trial
                    best_ms = ms
                    improved = True
    
    return best_order


def _multi_start_local_search(processing_times, initial_solutions, time_limit=30.0):
    """Run local search on multiple initial solutions and return the best."""
    start_time = time.time()
    best_order = None
    best_ms = float('inf')
    
    # First evaluate all initial solutions
    for sol in initial_solutions:
        ms = _makespan(processing_times, sol)
        if ms < best_ms:
            best_ms = ms
            best_order = list(sol)
    
    # Run local search on the best first
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1.0 and best_order is not None:
        best_order = _local_search(processing_times, best_order, time_limit=remaining)
        best_ms = _makespan(processing_times, best_order)
    
    # Then try other initial solutions if time permits
    for sol in initial_solutions:
        if sol == best_order:
            continue
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        candidate = _local_search(processing_times, sol, time_limit=remaining)
        ms = _makespan(processing_times, candidate)
        if ms < best_ms:
            best_ms = ms
            best_order = candidate
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multi-start heuristic (NEH, Palmer, LPT, SPT, insertion) followed by local search.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
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
    
    # Insertion heuristic applied to LPT order
    initial_solutions.append(_insertion_heuristic(processing_times, _lpt(processing_times)))
    
    # Insertion heuristic applied to SPT order
    initial_solutions.append(_insertion_heuristic(processing_times, _spt(processing_times)))
    
    # Add a few random permutations
    rng = random.Random(42)
    for _ in range(5):
        perm = list(range(n_jobs))
        rng.shuffle(perm)
        initial_solutions.append(perm)
    
    # Run multi-start local search
    best_order = _multi_start_local_search(processing_times, initial_solutions, time_limit=140.0)
    
    # If we still have time, try 2-opt on the best solution
    remaining = deadline - (time.time() - start_time)
    if remaining > 5.0 and best_order is not None:
        candidate = _two_opt(processing_times, best_order, time_limit=remaining - 5.0)
        ms_candidate = _makespan(processing_times, candidate)
        ms_best = _makespan(processing_times, best_order)
        if ms_candidate < ms_best:
            best_order = candidate
    
    return list(best_order)
# EVOLVE-BLOCK-END
