"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m - 1] if m > 0 else 0) + processing_times[job, m]
    return completion[-1]


def _makespan_vectorized(processing_times, order):
    """Vectorized makespan computation."""
    n_machines = processing_times.shape[1]
    # completion[m] tracks the completion time on machine m
    # For each job in order, completion[m] = max(completion[m], completion[m-1]) + pt[job, m]
    # We can compute this iteratively but vectorize over jobs using the sequential dependency
    # Actually the dependency is sequential across machines, so we do it per-job
    n_jobs = len(order)
    result = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        prev = 0.0
        for m in range(n_machines):
            val = max(prev, result[m]) + processing_times[job, m]
            result[m] = val
            prev = val
    return result[-1]


def _neh_initialization(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs = processing_times.shape[0]
    if n_jobs <= 2:
        return list(range(n_jobs))
    
    n_machines = processing_times.shape[1]
    totals = processing_times.sum(axis=1)
    # Sort by total processing time descending
    sorted_jobs = np.argsort(-totals)
    
    # Insert each job in best position
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order


def _adjacent_swap_delta(processing_times, order, i):
    """Compute makespan change if we swap order[i] and order[i+1]."""
    n_jobs = len(order)
    if i >= n_jobs - 1:
        return float('inf')
    
    # Current makespan
    current_ms = _makespan(processing_times, order)
    
    # Swapped order
    swapped = order.copy()
    swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
    swapped_ms = _makespan(processing_times, swapped)
    
    return swapped_ms - current_ms


def _improve_with_local_search(processing_times, order, time_limit):
    """Improve solution using adjacent swap local search with time limit."""
    start_time = time.time()
    n_jobs = len(order)
    current_ms = _makespan(processing_times, order)
    best_order = order.copy()
    best_ms = current_ms
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            delta = _adjacent_swap_delta(processing_times, order, i)
            if delta < -1e-9:
                # Accept the swap
                order[i], order[i+1] = order[i+1], order[i]
                current_ms += delta
                improved = True
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = order.copy()
        if time.time() - start_time > time_limit:
            break
    
    return best_order, best_ms


def _simulated_annealing(processing_times, order, time_limit):
    """Simulated annealing to escape local optima."""
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs < 3:
        return order, _makespan(processing_times, order)
    
    current_ms = _makespan(processing_times, order)
    best_order = order.copy()
    best_ms = current_ms
    
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    min_T = 1e-6
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
        
        # Random swap of two positions
        i = rng.integers(0, n_jobs - 1)
        j = rng.integers(i + 1, n_jobs)
        
        new_order = order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_ms = _makespan(processing_times, new_order)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            order = new_order
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = order.copy()
        
        T *= cooling
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs = processing_times.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Overall deadline guard: 150s limit, use 140s safety margin
    overall_start = time.time()
    overall_deadline = 140.0
    
    # Phase 1: NEH initialization (fast)
    neh_order = _neh_initialization(processing_times)
    neh_ms = _makespan(processing_times, neh_order)
    
    best_order = neh_order
    best_ms = neh_ms
    
    elapsed = time.time() - overall_start
    remaining = overall_deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Phase 2: Local search on NEH solution
    ls_time = min(remaining * 0.4, 30.0)
    ls_order, ls_ms = _improve_with_local_search(processing_times, neh_order, ls_time)
    if ls_ms < best_ms:
        best_ms = ls_ms
        best_order = ls_order
    
    elapsed = time.time() - overall_start
    remaining = overall_deadline - elapsed
    if remaining < 5.0:
        return best_order
    
    # Phase 3: Simulated annealing to escape local optima
    sa_time = min(remaining * 0.5, 40.0)
    sa_order, sa_ms = _simulated_annealing(processing_times, best_order, sa_time)
    if sa_ms < best_ms:
        best_ms = sa_ms
        best_order = sa_order
    
    elapsed = time.time() - overall_start
    remaining = overall_deadline - elapsed
    if remaining < 3.0:
        return best_order
    
    # Phase 4: Final local search polish
    polish_time = min(remaining * 0.7, 20.0)
    polish_order, polish_ms = _improve_with_local_search(processing_times, best_order, polish_time)
    if polish_ms < best_ms:
        best_ms = polish_ms
        best_order = polish_order
    
    return best_order
# EVOLVE-BLOCK-END
