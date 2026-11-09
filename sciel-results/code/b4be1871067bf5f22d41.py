# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, P):
    """Compute makespan for a given job order on processing_times matrix P."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_vectorized(order, P):
    """Vectorized makespan computation using cumulative sums."""
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    # For each machine, compute the completion time
    # completion[m] = sum_{i=0}^{m} max_{k=0}^{i} (sum_{j=0}^{k} P[order[j], j] - sum_{j=0}^{k-1} P[order[j], m])
    # This is complex; let's use a simpler approach
    
    # Actually, let's use the standard formula:
    # C(n, m) = max(C(n-1, m), C(n, m-1)) + P(n, m)
    # where C(n, m) is completion of job n on machine m
    
    # For a permutation, we can compute this efficiently
    n_jobs = len(order)
    # C[i][m] = completion of job i (in permutation order) on machine m
    # C[0][m] = sum_{k=0}^{m} P[order[0], k]
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[order[i], m]
    
    # We only need the last row's last element
    # Let's compute iteratively but in a somewhat vectorized way
    
    # For each job in order, maintain the completion times on each machine
    completion = np.zeros(n_machines, dtype=np.float64)
    for j in order:
        # For machine m, completion[m] = max(completion[m], completion[m-1]) + P[j, m]
        # But we need to be careful: completion[m-1] here refers to the new value for machine m-1
        # So we need to update sequentially
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + P[j, m]
            completion[m] = prev
    return float(completion[-1])


def _neh_heuristic(P):
    """NEH heuristic: build permutation by inserting each job at best position."""
    n_jobs = P.shape[0]
    if n_jobs == 0:
        return []
    
    totals = P.sum(axis=1)
    # Sort jobs by descending total processing time
    job_order = np.argsort(-totals)
    
    permutation = [job_order[0]]
    
    for idx in range(1, n_jobs):
        job = job_order[idx]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(permutation) + 1):
            # Try inserting job at position pos
            new_perm = permutation[:pos] + [job] + permutation[pos:]
            ms = _makespan_vectorized(new_perm, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        permutation = permutation[:best_pos] + [job] + permutation[best_pos:]
    
    return permutation


def _two_opt_improve(order, P, max_iterations=100):
    """Improve permutation using 2-opt local search."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_makespan = _makespan_vectorized(best_order, P)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Reverse segment i..j
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                ms = _makespan_vectorized(new_order, P)
                if ms < best_makespan - 1e-9:
                    best_order = new_order
                    best_makespan = ms
                    improved = True
        if not improved:
            break
    
    return best_order


def _insertion_improve(order, P, max_iterations=50):
    """Improve permutation using insertion local search."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_makespan = _makespan_vectorized(best_order, P)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n_jobs):
            job = best_order[i]
            remaining = best_order[:i] + best_order[i+1:]
            for pos in range(len(remaining) + 1):
                new_order = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan_vectorized(new_order, P)
                if ms < best_makespan - 1e-9:
                    best_order = new_order
                    best_makespan = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    P = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = P.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Start with NEH heuristic
    best_order = _neh_heuristic(P)
    best_makespan = _makespan_vectorized(best_order, P)
    
    # Check time
    if time.time() > deadline:
        return best_order
    
    # Try 2-opt improvement
    improved_order = _two_opt_improve(best_order, P, max_iterations=50)
    improved_makespan = _makespan_vectorized(improved_order, P)
    if improved_makespan < best_makespan:
        best_order = improved_order
        best_makespan = improved_makespan
    
    if time.time() > deadline:
        return best_order
    
    # Try insertion improvement
    improved_order = _insertion_improve(best_order, P, max_iterations=30)
    improved_makespan = _makespan_vectorized(improved_order, P)
    if improved_makespan < best_makespan:
        best_order = improved_order
        best_makespan = improved_makespan
    
    # If we still have time, do more iterations
    while time.time() < deadline - 5.0:
        # Alternate between 2-opt and insertion
        improved_order = _two_opt_improve(best_order, P, max_iterations=20)
        improved_makespan = _makespan_vectorized(improved_order, P)
        if improved_makespan < best_makespan:
            best_order = improved_order
            best_makespan = improved_makespan
        else:
            break
        
        if time.time() > deadline - 5.0:
            break
        
        improved_order = _insertion_improve(best_order, P, max_iterations=20)
        improved_makespan = _makespan_vectorized(improved_order, P)
        if improved_makespan < best_makespan:
            best_order = improved_order
            best_makespan = improved_makespan
        else:
            break
    
    return best_order
# EVOLVE-BLOCK-END
