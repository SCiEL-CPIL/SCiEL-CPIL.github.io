"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(pt, perm):
    """Compute makespan for a given permutation."""
    n_machines = pt.shape[1]
    machine_times = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            if m == 0:
                machine_times[m] += pt[job, m]
            else:
                machine_times[m] = max(machine_times[m - 1], machine_times[m]) + pt[job, m]
    return machine_times[-1]


def _makespan_fast(pt, perm):
    """Faster makespan computation using cumulative sums."""
    n_machines = pt.shape[1]
    # For each machine, compute the completion time
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + p[j][m]
    # We can vectorize partially
    
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0
    
    # perm is a list of job indices
    # Get the processing times in the order of perm
    pt_ordered = pt[perm, :]  # shape (n_jobs, n_machines)
    
    # Initialize machine completion times
    # We process jobs one by one, but we can vectorize across machines for each job
    # Actually, the recurrence is sequential in jobs, so let's do it efficiently
    
    # For machine 0, it's just cumulative sum
    # For other machines, it's more complex
    
    # Let's use the standard algorithm but with numpy where possible
    # Actually for small n_machines, the loop is fine
    
    completion = np.zeros(n_machines)
    for j in range(n_jobs):
        for m in range(n_machines):
            if m == 0:
                completion[0] += pt_ordered[j, 0]
            else:
                completion[m] = max(completion[m - 1], completion[m]) + pt_ordered[j, m]
    return completion[-1]


def _neh(pt):
    """NEH heuristic: sort by total processing time descending, then insert each job in best position."""
    n_jobs, n_machines = pt.shape
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build the permutation using NEH insertion
    perm = []
    for job in order:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(perm) + 1):
            # Try inserting job at position pos
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(pt, new_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        perm.insert(best_pos, job)
    
    return perm


def _palmer(pt):
    """Palmer heuristic: sort by sum of (n-1-2m)*p_jm for each job."""
    n_jobs, n_machines = pt.shape
    weights = (n_machines - 1 - 2 * np.arange(n_machines))
    scores = pt @ weights
    return list(np.argsort(-scores))


def _lpt(pt):
    """Longest Processing Time first."""
    return list(np.argsort(-pt.sum(axis=1)))


def _spt(pt):
    """Shortest Processing Time first."""
    return list(np.argsort(pt.sum(axis=1)))


def _random_perm(n_jobs):
    """Generate a random permutation."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _local_search(pt, perm, max_iterations=1000, time_limit=5.0):
    """2-opt local search: try swapping pairs of adjacent jobs."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _makespan_fast(pt, best_perm)
    
    start_time = time.time()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iteration += 1
        
        for i in range(n_jobs - 1):
            # Try swapping adjacent jobs i and i+1
            new_perm = list(best_perm)
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
            ms = _makespan_fast(pt, new_perm)
            if ms < best_makespan:
                best_perm = new_perm
                best_makespan = ms
                improved = True
    
    return best_perm, best_makespan


def _segment_reversal(pt, perm, max_iterations=500, time_limit=5.0):
    """Local search with segment reversal."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _makespan_fast(pt, best_perm)
    
    start_time = time.time()
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time.time() - start_time > time_limit:
            break
        improved = False
        iteration += 1
        
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                new_perm = list(best_perm)
                new_perm[i:j+1] = reversed(new_perm[i:j+1])
                ms = _makespan_fast(pt, new_perm)
                if ms < best_makespan:
                    best_perm = new_perm
                    best_makespan = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_perm, best_makespan


def order_jobs(processing_times):
    """
    Order jobs to minimize makespan on a flow shop.
    
    Uses multiple heuristics and local search with a time budget.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    time_budget = 100.0  # Leave margin for safety
    
    # Generate initial solutions using different heuristics
    candidates = []
    
    # NEH
    candidates.append(_neh(pt))
    
    # Palmer
    candidates.append(_palmer(pt))
    
    # LPT
    candidates.append(_lpt(pt))
    
    # SPT
    candidates.append(_spt(pt))
    
    # Random permutations
    for _ in range(5):
        candidates.append(_random_perm(n_jobs))
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_makespan = float('inf')
    
    for perm in candidates:
        ms = _makespan_fast(pt, perm)
        if ms < best_makespan:
            best_makespan = ms
            best_perm = perm
    
    # Apply local search to the best candidate
    if time.time() - start_time < time_budget - 10.0:
        best_perm, best_makespan = _local_search(pt, best_perm, max_iterations=2000, time_limit=time_budget - 15.0)
    
    # Try segment reversal on the result
    if time.time() - start_time < time_budget - 5.0:
        alt_perm, alt_makespan = _segment_reversal(pt, best_perm, max_iterations=500, time_limit=time_budget - 5.0)
        if alt_makespan < best_makespan:
            best_perm = alt_perm
            best_makespan = alt_makespan
    
    # If we still have time, do more random restarts
    while time.time() - start_time < time_budget - 2.0:
        random_perm = _random_perm(n_jobs)
        ms = _makespan_fast(pt, random_perm)
        if ms < best_makespan:
            best_makespan = ms
            best_perm = random_perm
            # Local search from this new best
            if time.time() - start_time < time_budget - 3.0:
                best_perm, best_makespan = _local_search(pt, best_perm, max_iterations=1000, time_limit=time_budget - 3.0)
    
    # Safety check: ensure we return a valid permutation
    if best_perm is None:
        best_perm = list(range(n_jobs))
    
    return best_perm
# EVOLVE-BLOCK-END
