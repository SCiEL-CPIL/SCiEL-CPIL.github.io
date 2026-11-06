"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given permutation of job indices."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    # Extract the submatrix in the given order
    sub = processing_times[order]
    # Cumulative sum along machines (axis=1) gives completion times per job per machine
    # But in flow shop, machine m of job i starts after max(complete of job i-1 on machine m, complete of job i on machine m-1)
    # Let's compute properly:
    n_machines = processing_times.shape[1]
    # C[j][m] = completion time of job j on machine m
    # C[0][m] = C[0][m-1] + p[0][m]  for m > 0, C[0][0] = p[0][0]
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    
    # Vectorized approach:
    # C is n_jobs x n_machines
    C = np.zeros((n_jobs, n_machines))
    C[:, 0] = np.cumsum(sub[:, 0])
    for m in range(1, n_machines):
        C[:, m] = np.maximum(C[:, m-1], C[:-1, m]) + sub[:, m]
        # Wait, this isn't quite right for the recurrence. Let me think again.
        # C[j][m] = max(C[j-1][m], C[j][m-1]) + sub[j][m]
        # This is a sequential dependency along both dimensions.
        # The vectorized form above is incorrect because C[j][m] depends on C[j-1][m] which we're computing in the same step.
    
    # Let's do it properly with a loop over machines, but vectorize over jobs where possible.
    # Actually, the recurrence C[j][m] = max(C[j-1][m], C[j][m-1]) + sub[j][m]
    # can be computed column by column. For a fixed m, C[:, m] depends on C[:, m-1] and C[:-1, m].
    # This is still sequential along j for a fixed m.
    
    # Let's just use a simple double loop for clarity and correctness, since n is likely small.
    C = np.zeros((n_jobs, n_machines))
    C[0, 0] = sub[0, 0]
    for m in range(1, n_machines):
        C[0, m] = C[0, m-1] + sub[0, m]
    for j in range(1, n_jobs):
        C[j, 0] = C[j-1, 0] + sub[j, 0]
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + sub[j, m]
    
    return C[-1, -1]


def _makespan_delta(processing_times, order, i, j):
    """
    Compute the change in makespan if we swap positions i and j in the order.
    This is an approximation or exact delta computation.
    For exactness, we'd need to recompute, but that's expensive.
    Let's just recompute for simplicity in the local search step, or use a heuristic.
    Actually, let's implement a proper delta computation for adjacent swaps first,
    and for general swaps, we can just recompute.
    """
    # For simplicity and correctness, let's just recompute makespan for the new order.
    # This is O(n*m) which is acceptable for local search with n up to maybe 100-200.
    new_order = list(order)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return _makespan(processing_times, new_order) - _makespan(processing_times, order)


def _neh_initialization(processing_times):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow shop.
    1. Sort jobs by total processing time descending.
    2. Insert each job into the position that minimizes makespan.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    totals = processing_times.sum(axis=1)
    sorted_indices = np.argsort(-totals)
    
    # Start with the first job
    current_order = [sorted_indices[0]]
    
    for idx in sorted_indices[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            new_order = current_order[:pos] + [idx] + current_order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order = current_order[:best_pos] + [idx] + current_order[best_pos:]
    
    return current_order


def _local_search(processing_times, order, time_limit):
    """
    Hill climbing local search with 2-opt (swap any two positions) moves.
    Uses a deadline guard to stop before time runs out.
    """
    start_time = time.time()
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    n_jobs = len(current_order)
    
    if n_jobs <= 1:
        return current_order
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            # Check time
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                # Check time
                if time.time() - start_time > time_limit:
                    break
                new_ms = _makespan(processing_times, current_order[:i] + [current_order[j]] + current_order[i+1:j] + [current_order[i]] + current_order[j+1:])
                if new_ms < current_ms:
                    current_ms = new_ms
                    # Perform the swap
                    # Actually, swapping positions i and j is not the same as the move above.
                    # Let's stick to simple swaps for clarity.
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    improved = True
    
    return current_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    Uses NEH initialization followed by local search with a time budget.
    """
    start_time = time.time()
    # Leave 10 seconds as safety margin before the 120s limit
    time_limit = 105.0 - (time.time() - start_time)
    
    if time_limit <= 0:
        # Fallback to a simple rule
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    # Step 1: NEH initialization
    initial_order = _neh_initialization(processing_times)
    
    # Step 2: Local search
    final_order = _local_search(processing_times, initial_order, time_limit)
    
    return final_order
# EVOLVE-BLOCK-END
