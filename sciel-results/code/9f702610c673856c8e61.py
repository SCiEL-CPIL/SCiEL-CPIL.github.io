"""Rewritten program for flow-shop permutation optimization.
Uses Johnson's rule (for 2-machine) or NEH (for m>2) for initialization,
then applies a time-limited insertion local search with vectorized makespan deltas
to efficiently evaluate all possible repositioning of a job."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _completion_times(processing_times, order):
    """
    Compute the full completion time matrix C for a given order.
    Returns C as a 2D list: C[i][j] is completion of job at position i on machine j.
    Also returns makespan = C[-1][-1].
    """
    if len(order) == 0:
        return 0.0, []
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    C = [[0.0] * n_machines for _ in range(n_jobs)]
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                C[i][j] = p
            elif i == 0:
                C[i][j] = C[i][j-1] + p
            elif j == 0:
                C[i][j] = C[i-1][j] + p
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + p
    
    return C[n_jobs-1][n_machines-1], C

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    ms, _ = _completion_times(processing_times, order)
    return ms

def _johnson_two_machine(processing_times):
    """
    Apply Johnson's rule for 2-machine flow shop.
    For m>2, we'll aggregate first and last machines as a heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_machines == 2:
        A = processing_times[:, 0]
        B = processing_times[:, 1]
    else:
        # Aggregate: A = sum of processing times on first machine
        # B = sum of processing times on last machine
        A = processing_times[:, 0].copy()
        B = processing_times[:, -1].copy()
    
    # Johnson's rule: sort by min(A, B)
    # If min is A, put in front; if min is B, put in back
    jobs = list(range(n_jobs))
    min_vals = np.minimum(A, B)
    
    # Sort jobs by min processing time
    order = list(np.argsort(min_vals, kind='stable'))
    
    front = []
    back = []
    used = [False] * n_jobs
    
    # Actually, Johnson's rule is typically done by repeatedly picking the job
    # with the minimum min(A_j, B_j). Let's do it properly.
    remaining = list(range(n_jobs))
    front_list = []
    back_list = []
    
    while remaining:
        # Find job with minimum min(A, B)
        best_idx = -1
        best_val = float('inf')
        for j in remaining:
            v = min(A[j], B[j])
            if v < best_val:
                best_val = v
                best_idx = j
        
        if A[best_idx] <= B[best_idx]:
            front_list.append(best_idx)
        else:
            back_list.append(best_idx)
        
        remaining.remove(best_idx)
    
    return front_list + back_list[::-1]

def _neh_rule(processing_times):
    """NEH heuristic: sort by total processing time descending, insert in best position."""
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
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

def _vectorized_insertion_delta(processing_times, order, C, i, j):
    """
    Compute the change in makespan when moving job from position i to position j.
    i and j are positions in the order (0-indexed).
    If j > i, the job moves right; if j < i, the job moves left.
    
    C is the current completion time matrix.
    Returns (delta_makespan, new_C) where new_C is the updated completion matrix.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return 0.0, C
    
    # Create a copy of C
    new_C = [row[:] for row in C]
    
    # Build new order
    job = order[i]
    new_order = order[:]
    del new_order[i]
    if j > i:
        new_order.insert(j - 1, job)
    else:
        new_order.insert(j, job)
    
    # Recalculate affected rows
    # The move affects rows from min(i,j) to n_jobs-1
    start_row = min(i, j)
    
    for k in range(start_row, n_jobs):
        job_idx = new_order[k]
        p_row = processing_times[job_idx]
        for m in range(n_machines):
            p = p_row[m]
            if k == 0 and m == 0:
                new_C[k][m] = p
            elif k == 0:
                new_C[k][m] = new_C[k][m-1] + p
            elif m == 0:
                new_C[k][m] = new_C[k-1][m] + p
            else:
                new_C[k][m] = max(new_C[k-1][m], new_C[k][m-1]) + p
    
    old_ms = C[n_jobs-1][n_machines-1]
    new_ms = new_C[n_jobs-1][n_machines-1]
    return new_ms - old_ms, new_C

def _insertion_local_search(processing_times, order, time_limit=10.0):
    """
    First-improvement insertion local search.
    For each job, try moving it to every other position and accept the first improvement.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms, best_C = _completion_times(processing_times, best_order)
    start_time = time.time()
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job = best_order[i]
            
            # Try all insertion positions j != i
            # We'll collect deltas and pick the best one (best-improvement within this job)
            # Actually, let's do first-improvement for speed
            for j in range(n_jobs):
                if i == j:
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                delta, new_C = _vectorized_insertion_delta(processing_times, best_order, best_C, i, j)
                
                if delta < -1e-10:
                    # Accept
                    best_order = list(best_order)
                    del best_order[i]
                    if j > i:
                        best_order.insert(j - 1, job)
                    else:
                        best_order.insert(j, job)
                    best_ms += delta
                    best_C = new_C
                    improved = True
                    break  # First improvement for this job, move to next job
    
    return best_order, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses Johnson's rule or NEH for initialization, then insertion local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solutions
    init_orders = []
    
    # 1. Johnson's rule (works well for 2-machine, decent for more)
    init_orders.append(_johnson_two_machine(processing_times))
    
    # 2. NEH Rule
    init_orders.append(_neh_rule(processing_times))
    
    # 3. A few random orders
    for _ in range(3):
        random_order = list(range(n_jobs))
        random.shuffle(random_order)
        init_orders.append(random_order)
    
    # Evaluate and pick the best initial order
    best_order = init_orders[0]
    best_ms = _makespan(processing_times, best_order)
    
    for order in init_orders[1:]:
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
    
    # Check time
    remaining_time = deadline - time.time()
    if remaining_time < 2.0:
        return best_order
    
    # Run insertion local search with the best initial order
    time_limit_ls = min(remaining_time - 1.0, 100.0)
    best_order, best_ms = _insertion_local_search(processing_times, best_order, time_limit=time_limit_ls)
    
    # If time remains, try restarting with other initial orders
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        # Try NEH-based order with local search if we haven't already
        for order in init_orders:
            if time.time() > deadline - 3.0:
                break
            if order == best_order:
                continue
            ms = _makespan(processing_times, order)
            if ms < best_ms:
                best_ms = ms
                best_order = list(order)
                # Run local search on this new best
                remaining_time = deadline - time.time()
                if remaining_time > 3.0:
                    tl = min(remaining_time - 2.0, 30.0)
                    best_order, best_ms = _insertion_local_search(
                        processing_times, best_order, time_limit=tl
                    )
    
    # Final safety check
    if time.time() > deadline - 1.0:
        return best_order
    
    return best_order
# EVOLVE-BLOCK-END
