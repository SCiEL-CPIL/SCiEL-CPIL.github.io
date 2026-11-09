"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """
    Compute makespan for a given permutation using vectorized operations over machines.
    perm: list or array of job indices.
    pt: numpy array of shape (n_jobs, n_machines).
    """
    if len(perm) == 0:
        return 0.0
    
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # completion[m] holds the completion time of the last processed job on machine m
    # Initially 0 for all machines
    completion = np.zeros(n_machines)
    
    # Get processing times for all jobs in the order of perm
    # p_rows is shape (n_jobs, n_machines)
    p_rows = pt[perm]
    
    for i in range(n_jobs):
        p = p_rows[i]
        # For machine 0: completion[0] += p[0]
        # For machine k > 0: completion[k] = max(completion[k], completion[k-1]) + p[k]
        # We can do this in a loop over machines, but since n_machines is usually small, 
        # or we can vectorize? 
        # Vectorization is tricky because of the dependency on completion[k-1].
        # However, we can update sequentially.
        
        # Optimization: Use a local variable for the previous completion time
        prev_c = 0.0
        for k in range(n_machines):
            if k == 0:
                c = completion[0] + p[0]
            else:
                c = max(completion[k], prev_c) + p[k]
            completion[k] = c
            prev_c = c
            
    return completion[-1]

def _neh(processing_times):
    """
    Construct initial solution using NEH heuristic.
    """
    n, m = processing_times.shape
    if n == 0:
        return []
    
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    current_perm = []
    for job in order:
        best_pos = 0
        best_mk = float('inf')
        
        # Try all insertion positions
        for pos in range(len(current_perm) + 1):
            # Create new permutation
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            mk = _makespan(new_perm, processing_times)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def order_jobs(processing_times):
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
        
    start_time = time.time()
    # Safety margin: 120s limit, leave 5s for overhead
    deadline = start_time + 115.0
    
    # 1. Initial Solution: NEH
    current_perm = _neh(processing_times)
    current_mk = _makespan(current_perm, processing_times)
    
    best_perm = current_perm[:]
    best_mk = current_mk
    
    # 2. Local Search: Insertion Moves
    # We iterate until time runs out.
    # In each step, we try to improve the current solution by moving a job to a better position.
    # To avoid getting stuck in local minima, we can use a simple random restart or simulated annealing logic.
    # Given the time budget, a simple "best improvement" or "first improvement" local search 
    # with random restarts if stuck is often effective.
    # However, pure local search can get stuck. Let's use a simple Simulated Annealing 
    # but with INSERTION moves instead of SWAP, as insertion is more powerful.
    
    # Parameters
    T = 1.0 * (processing_times.sum() / max(1, n)) # Scale temperature to average job size
    cooling = 0.99
    min_T = 1e-6
    
    n_jobs = len(current_perm)
    
    # Helper to check time
    def _time_left():
        return time.time() < deadline

    while _time_left():
        if T < min_T:
            # Restart if needed? Or just break.
            # Let's break and return best.
            break
            
        # Generate neighbor: Move a random job to a random position (Insertion)
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        
        # Remove job from position i
        # If we remove it, positions shift.
        # Let's pick a target position j in the range [0, n_jobs-1]
        # Note: if we insert at position j, and i < j, the job effectively moves to j-1 in the new list?
        # Let's just construct the new list explicitly to be safe and clear, though slightly slower.
        # For n < 1000, this is fine.
        
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        # Create candidate
        # Remove job at i
        candidate = current_perm[:i] + current_perm[i+1:]
        # Insert at j
        # If i < j, the target index in the shortened list is j-1?
        # Actually, standard insertion:
        # If we move element at i to position j in the ORIGINAL list indices:
        # It's easier to just pick a random index in the new list.
        # Let's pick a random index k in [0, n_jobs-1] for the new position.
        # Then insert job at k.
        
        # Let's restart the neighbor generation logic for clarity:
        # Pick job index i.
        # Pick insertion index k in [0, n_jobs-1].
        # Remove job at i.
        # Insert job at k in the resulting list.
        
        # The above logic with i and j was a bit ambiguous. Let's do:
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        
        # Remove
        new_list = current_perm[:i] + current_perm[i+1:]
        
        # Pick insertion point in the new list (length n_jobs-1)
        # Valid indices for insertion in a list of length L are 0..L
        k = random.randint(0, n_jobs - 1)
        
        candidate = new_list[:k] + [job] + new_list[k:]
        
        # Evaluate
        cand_mk = _makespan(candidate, processing_times)
        delta = cand_mk - current_mk
        
        # SA Acceptance
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            current_perm = candidate
            current_mk = cand_mk
            
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm
# EVOLVE-BLOCK-END
