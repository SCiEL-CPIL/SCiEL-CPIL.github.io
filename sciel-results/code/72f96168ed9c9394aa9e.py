"""Improved program for flow-shop permutation construction using NEH + Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = P.shape[1]
    
    # C[m] is the completion time of the last processed job on machine m
    # For each job j in order, completion on machine m is:
    #   C_new[m] = max(C[m], C_new[m-1]) + P[j, m]
    # We can vectorize over jobs but the sequential dependency on machines remains.
    # For moderate n_jobs and n_machines, a loop is fine.
    
    C = np.zeros(n_machines)
    for j in order:
        # For this job j, compute completion times across machines
        # C_new[m] = max(C[m], C_new[m-1]) + P[j, m]
        # This is sequential over m, so we do a loop over machines
        prev = 0
        for m in range(n_machines):
            val = max(C[m], prev) + P[j, m]
            C[m] = val
            prev = val
    
    return C[-1]


def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow shop.
    Sort jobs by total processing time (descending), then insert each job
    at the best position in the current partial sequence.
    """
    n_jobs, n_machines = P.shape
    
    # Sort by total processing time, descending
    totals = P.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Build sequence greedily
    seq = [order[0]]
    for idx in range(1, n_jobs):
        job = order[idx]
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(seq) + 1):
            # Insert job at position pos
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _two_opt(order, P, max_iterations=1000):
    """
    Simple 2-opt local search: try swapping adjacent jobs or reversing segments.
    We'll do a simple adjacent swap local search for efficiency.
    """
    n = len(order)
    current_order = list(order)
    current_ms = _makespan(current_order, P)
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            # Try swapping adjacent jobs i and i+1
            new_order = current_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            new_ms = _makespan(new_order, P)
            if new_ms < current_ms:
                current_order = new_order
                current_ms = new_ms
                improved = True
                # Restart from the beginning to find more improvements
                break
    
    return current_order, current_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic followed by local search with a time deadline.
    """
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    P = np.asarray(processing_times)
    n_jobs = P.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: NEH heuristic
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    # Phase 2: Local search (adjacent swaps)
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:  # leave 5s safety margin
        current_order, current_ms = _two_opt(best_order, P)
        if current_ms < best_ms:
            best_order = current_order
            best_ms = current_ms
    
    # Phase 3: Additional restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        # Try a few random restarts with perturbation
        rng = np.random.default_rng(42)
        for restart in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline - 5:
                break
            
            # Perturb the best solution: random swap of two positions
            perturbed = list(best_order)
            i, j = rng.choice(n_jobs, size=2, replace=False)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            # Local search from perturbed solution
            local_order, local_ms = _two_opt(perturbed, P)
            if local_ms < best_ms:
                best_order = local_order
                best_ms = local_ms
    
    return list(best_order)
# EVOLVE-BLOCK-END
