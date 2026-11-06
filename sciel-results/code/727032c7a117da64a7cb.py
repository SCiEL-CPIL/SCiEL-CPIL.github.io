"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    Optimized for speed using local variables and minimal overhead.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # We will compute the completion time matrix C on the fly or just track the last row?
    # For flow shop, C[i, j] = max(C[i-1, j], C[i, j-1]) + P[i, j]
    # We only need the previous row to compute the current row.
    
    # To be safe and simple, we can compute the full matrix if N*M is small, 
    # but for larger N, we can just keep the previous row.
    # However, for delta evaluation, we need to recompute specific parts.
    # For the main loop, a simple O(N*M) loop is fine if N*M is small.
    
    # Let's use the row-by-row approach with numpy for speed?
    # Actually, pure Python loops are slow. 
    # Let's try to vectorize the row computation if possible?
    # C[i, :] depends on C[i-1, :] and C[i, :-1].
    # This is a scan operation.
    
    # For now, let's use a standard loop but optimized.
    # If N and M are small (e.g. < 50), this is fine.
    # If they are large, we need something better.
    
    # Let's assume N, M are moderate.
    # We will use a list of lists for C to avoid numpy overhead in tight loops if needed,
    # but numpy is usually faster for array ops.
    
    # Actually, the best pure-python/NumPy hybrid for small N,M is just loops.
    # Let's stick to a robust implementation.
    
    # C[i, j]
    # We can use a 2D array for C
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    for i in range(n_jobs):
        job = perm[i]
        p_row = processing_times[job]
        if i == 0:
            C[i, 0] = p_row[0]
            for j in range(1, n_machines):
                C[i, j] = C[i, j-1] + p_row[j]
        else:
            C[i, 0] = C[i-1, 0] + p_row[0]
            for j in range(1, n_machines):
                # max(C[i-1, j], C[i, j-1]) + p_row[j]
                prev_machine = C[i-1, j]
                prev_job = C[i, j-1]
                if prev_machine > prev_job:
                    C[i, j] = prev_machine + p_row[j]
                else:
                    C[i, j] = prev_job + p_row[j]
    
    return C[-1, -1]

def _delta_makespan(processing_times, perm, i, j):
    """
    Calculate the change in makespan if we swap jobs at positions i and j in perm.
    This is an approximation or exact?
    Exact delta evaluation for flow shop is complex.
    However, for small N, we can just recalculate the makespan.
    If N is small (< 20), _calc_makespan is fast enough for ILS.
    If N is large, we need a better delta.
    
    Given the 120s limit and typical instance sizes, let's assume N is small enough
    that O(N*M) recalculation is viable for thousands of iterations.
    If N=100, M=10, that's 1000 ops. 10,000 iterations = 10M ops. Python might be too slow.
    
    Let's check the size. If N*M is large, we might timeout.
    We will implement a check: if N*M > 1000, use a heuristic delta or fewer iterations.
    Actually, let's just use _calc_makespan and rely on the time guard.
    """
    pass # We will just use _calc_makespan for now in the loop, it's simpler and robust.

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = list(current_perm)
    best_ms = _calc_makespan(processing_times, best_perm)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    # 2. Iterated Local Search (ILS)
    # Neighborhood: Swap two jobs, or Move a job.
    # For simplicity and speed, we will use Swap (2-opt) and Move.
    
    # We will run local search (steepest descent) until no improvement, then perturb.
    
    def local_search(perm, ms):
        improved = True
        while improved:
            improved = False
            # Try all swaps
            for i in range(n_jobs):
                if time.time() > deadline - 5.0:
                    return perm, ms
                for j in range(i + 1, n_jobs):
                    # Swap i and j
                    perm[i], perm[j] = perm[j], perm[i]
                    new_ms = _calc_makespan(processing_times, perm)
                    if new_ms < ms:
                        ms = new_ms
                        improved = True
                    else:
                        # Swap back
                        perm[i], perm[j] = perm[j], perm[i]
        return perm, ms
    
    def perturb(perm, strength):
        """
        Randomly swap 'strength' pairs of jobs.
        """
        p = list(perm)
        for _ in range(strength):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                p[i], p[j] = p[j], p[i]
        return p
    
    # ILS Loop
    perturb_strength = 2
    
    while time.time() < deadline - 5.0:
        # Local Search
        current_perm, current_ms = local_search(current_perm, current_ms)
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        # Perturbation
        current_perm = perturb(best_perm, perturb_strength)
        
        # Optional: adapt perturb strength
        # If no improvement in last 100 iterations, increase strength?
        # For now, keep it simple.
        
    return best_perm
# EVOLVE-BLOCK-END
