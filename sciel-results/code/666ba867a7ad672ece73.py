"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation using a vectorized NumPy approach.
    This avoids Python-level inner loops over machines.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    # We process jobs one by one, but update all machines at once using numpy.
    # completion[m] is the completion time of the last job on machine m.
    completion = np.zeros(n_machines)
    
    for j in perm:
        row = processing_times[j]
        
        # For the first machine: C_0 = C_0 + P_0
        # For subsequent machines: C_m = max(C_m, C_{m-1}) + P_m
        # We can compute the new completion times for all m >= 1 in a vectorized way
        # if we can handle the dependency on C_{m-1}.
        
        # Actually, the dependency C_m depends on C_{m-1} (current job) means we can't
        # fully vectorize across machines in one go for the max operation without
        # a scan (cumulative max). However, we can use a simple loop for the machines
        # if n_machines is small, or use a different trick.
        
        # Let's try a hybrid: if n_machines is small, loop. If large, maybe numpy is better?
        # Actually, for flow shop, the dependency chain is strict:
        # new_c[0] = completion[0] + row[0]
        # new_c[1] = max(completion[1], new_c[0]) + row[1]
        # ...
        
        # This is a sequential dependency. We cannot fully vectorize the machine loop
        # without using a specialized scan operation which might not be faster in Python
        # for small n_machines.
        
        # However, we can vectorize the *job* loop? No, jobs are sequential too.
        
        # Let's stick to a tight Python loop for the machines but ensure it's as fast as possible.
        # Or, we can use a list of floats for completion to avoid numpy array indexing overhead?
        
        # Let's try a pure Python list for completion times, which is often faster for small n.
        comp = [0.0] * n_machines
        for j in perm:
            row = processing_times[j]
            prev_c = 0.0
            for m in range(n_machines):
                if m == 0:
                    new_c = comp[0] + row[0]
                else:
                    if comp[m] > prev_c:
                        new_c = comp[m] + row[m]
                    else:
                        new_c = prev_c + row[m]
                comp[m] = new_c
                prev_c = new_c
        return float(comp[-1])


def _calc_makespan_fast(processing_times, perm):
    """
    Optimized makespan calculation using a list for completion times to reduce overhead.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
    if n_machines == 1:
        return float(processing_times[perm, 0].sum())
    
    comp = [0.0] * n_machines
    pt = processing_times
    nm = n_machines
    
    for j in perm:
        row = pt[j]
        prev_c = 0.0
        for m in range(nm):
            if m == 0:
                new_c = comp[0] + row[0]
            else:
                if comp[m] > prev_c:
                    new_c = comp[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp[m] = new_c
            prev_c = new_c
    return float(comp[-1])


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
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
            ms = _calc_makespan_fast(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan_fast(processing_times, best_perm)
    
    # 2. Simulated Annealing
    # Parameters
    T = max(1.0, best_ms * 0.01)  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.9995
    max_iterations = 100000  # Will be limited by time
    
    current_perm = best_perm
    current_ms = best_ms
    
    # Pre-compute some data if needed? No, just run SA.
    
    while time.time() < deadline - 2.0:
        # Generate a random neighbor
        # Move: pick two random positions i, j and swap
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        new_ms = _calc_makespan_fast(processing_times, new_perm)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Improvement
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm
        else:
            # Deterioration: accept with probability exp(-delta / T)
            if T > T_min:
                probability = np.exp(-delta / T)
                if random.random() < probability:
                    current_perm = new_perm
                    current_ms = new_ms
        
        # Cool down
        T *= cooling_rate
        
    return best_perm
# EVOLVE-BLOCK-END
