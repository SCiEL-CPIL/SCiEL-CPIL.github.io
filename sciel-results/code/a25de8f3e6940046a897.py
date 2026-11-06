"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    This implementation is optimized for small N (jobs) and M (machines).
    For flow shop, C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m].
    We iterate over jobs (outer) and machines (inner).
    Since M is usually small (e.g., 5-20), the inner loop is short.
    Using a list of floats for the current row completion times is faster than numpy arrays for very small M due to overhead.
    However, if M is large, numpy might be better. 
    Given typical flow shop instances, M is small.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    # perm is a list of job indices
    # processing_times is (N, M)
    
    # completion[m] holds the completion time of the last job on machine m
    # We initialize with zeros
    # Actually, we need to be careful with types.
    
    # Let's stick to a tight python loop but optimized.
    # Pre-extract rows? No, perm is dynamic.
    
    # For very small N, this is fine.
    # For larger N, we might need C extensions or better vectorization.
    # But let's try to keep it simple and robust.
    
    n_machines = processing_times.shape[1]
    
    # If n_machines is 1, makespan is sum of processing times of all jobs.
    if n_machines == 1:
        return np.sum(processing_times[perm, 0])
        
    # Standard loop
    # completion[m] for the current last job processed
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = processing_times[job_idx]
        prev_c = 0.0
        for m in range(n_machines):
            # c_prev = completion[m] (completion of previous job on this machine)
            # c_left = prev_c (completion of current job on previous machine)
            # new_c = max(c_prev, c_left) + row[m]
            if completion[m] > prev_c:
                new_c = completion[m] + row[m]
            else:
                new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    return completion[-1]

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic (Fast version)
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    # For NEH, we insert each job at the best position.
    # To make this faster, we can limit the search or just use it as a seed.
    # Since SA will do the heavy lifting, a good initial solution is important.
    # Let's do a simplified NEH: insert at best position among a few candidates or full search if N is small.
    
    # For N < 50, full NEH is fast enough.
    if n_jobs <= 50:
        for i in range(1, n_jobs):
            job = sorted_jobs[i]
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(current_perm) + 1):
                candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
                ms = _calc_makespan_vectorized(processing_times, candidate_perm)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    else:
        # For large N, just append in sorted order to save time, SA will fix it.
        # Or use a simpler heuristic.
        current_perm = sorted_jobs.tolist()
        
    best_perm = current_perm
    best_ms = _calc_makespan_vectorized(processing_times, best_perm)
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 1.0  # Initial temperature
    T_end = 0.001  # Final temperature
    cooling_rate = 0.995  # Cooling factor
    
    # We need to estimate the scale of makespan to set T_start appropriately.
    # A rough estimate: average processing time * N * M?
    # Or just use a relative scale.
    # Let's scale T_start based on the initial makespan.
    # Delta C is usually small relative to C.
    # If C is 1000, delta C might be 10. So T should start around 10-50.
    T_start = max(10.0, best_ms * 0.05)
    
    current_perm = best_perm
    current_ms = best_ms
    T = T_start
    
    # Neighborhood: 2-opt (reverse segment)
    # We will perform a fixed number of iterations or until time runs out.
    # Since we don't know N, we'll just loop until deadline.
    
    # To speed up, we can pre-calculate some things? No, flow shop is complex.
    
    # Optimization: Use a list for perm to allow fast slicing?
    # Slicing a list is O(N). Reversing is O(N).
    # _calc_makespan is O(N*M).
    
    # Let's use a numpy array for perm? No, list is faster for small N operations in Python if we avoid numpy overhead in the loop.
    # Actually, list slicing in Python is quite fast.
    
    while T > T_end and (time.time() < deadline - 2.0):
        
        # Generate a neighbor: 2-opt move
        i = random.randint(0, n_jobs - 2)
        j = random.randint(i + 1, n_jobs - 1)
        
        # Create new permutation
        # new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
        
        # To avoid creating a new list every time if possible?
        # We have to create it to pass to _calc_makespan.
        
        new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
        new_ms = _calc_makespan_vectorized(processing_times, new_perm)
        
        delta = new_ms - current_ms
        
        # Acceptance probability
        if delta < 0:
            # Better
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm
        else:
            # Worse
            # P = exp(-delta / T)
            # If delta is large and T is small, exp is 0.
            # Optimization: if delta > T * 10, probability is ~0.
            if delta > T * 10:
                pass # Reject
            else:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
        
        # Cool
        T *= cooling_rate
        
        # Optional: Random restarts?
        # If we get stuck, we can restart from best_perm with higher T.
        # But simple SA usually works well.
        
    return best_perm
# EVOLVE-BLOCK-END
