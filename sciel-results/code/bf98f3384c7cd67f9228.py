"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan_fast(processing_times, perm):
    """
    Fast makespan calculation.
    Uses a column-wise loop which is often faster for small M and large N
    due to better cache locality and fewer Python loop iterations if M < N.
    """
    if len(perm) == 0:
        return 0.0
    
    p = processing_times[perm]
    n_jobs, n_machines = p.shape
    
    if n_machines == 1:
        return np.sum(p)
    
    # C[i, j] = completion time of job i on machine j
    # C[:, 0] is cumulative sum of p[:, 0]
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    
    # We store the current column in a 1D array to save memory and improve cache
    # prev_col represents C[:, j-1]
    # curr_col represents C[:, j]
    
    prev_col = np.cumsum(p[:, 0])
    
    for j in range(1, n_machines):
        curr_col = np.empty(n_jobs, dtype=np.float64)
        c_prev_row = 0.0
        p_col = p[:, j]
        
        # This loop is sequential but tight.
        for i in range(n_jobs):
            # max(c_prev_row, prev_col[i]) + p_col[i]
            if c_prev_row > prev_col[i]:
                val = c_prev_row + p_col[i]
            else:
                val = prev_col[i] + p_col[i]
            curr_col[i] = val
            c_prev_row = val
            
        prev_col = curr_col
        
    return prev_col[-1]

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
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
    
    # 2. Iterated Local Search (ILS) with Or-opt and 2-opt
    # This approach is often superior to SA for combinatorial problems as it
    # focuses on escaping local optima via random perturbations followed by
    # rigorous local search, rather than relying on temperature schedules.
    
    current_perm = best_perm
    current_ms = best_ms
    
    # Local Search Function: Or-opt (move 1 or 2 jobs) + 2-opt (reverse segment)
    # We run this until no improvement is found or time is tight.
    def local_search(perm):
        perm = list(perm)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            
            # Try Or-opt: move a block of 1 or 2 jobs
            for block_size in [1, 2]:
                if block_size > n_jobs:
                    continue
                for i in range(n_jobs - block_size + 1):
                    if time.time() > deadline - 5.0:
                        return perm
                    
                    # Extract block
                    block = perm[i:i+block_size]
                    # Remove block
                    remaining = perm[:i] + perm[i+block_size:]
                    
                    # Try all insertion positions
                    for j in range(len(remaining) + 1):
                        if j == i: # Skip if inserting back at same place (roughly)
                            continue
                        # Actually, if we removed from i, inserting at i is the same position.
                        # If we removed from i, inserting at i+block_size is also same.
                        # But since we work with 'remaining', indices shift.
                        # Let's just check all j.
                        
                        new_perm = remaining[:j] + block + remaining[j:]
                        new_ms = _calc_makespan_fast(processing_times, new_perm)
                        
                        if new_ms < current_ms - 1e-9:
                            current_perm = new_perm
                            current_ms = new_ms
                            improved = True
                            # Restart local search with new perm?
                            # For simplicity, continue with this new perm in the loop
                            # But we need to update the 'perm' variable used in the loop
                            # It's better to return and restart, or just update and break loops
                            # Let's update and break to restart the while loop
                            return local_search(new_perm)
                            
            # Try 2-opt: reverse a segment
            for i in range(n_jobs - 1):
                if time.time() > deadline - 5.0:
                    return perm
                for j in range(i + 1, n_jobs):
                    new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                    new_ms = _calc_makespan_fast(processing_times, new_perm)
                    if new_ms < current_ms - 1e-9:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        return local_search(new_perm)
                        
        return current_perm
    
    # Main ILS Loop
    while time.time() < deadline - 5.0:
        # Local Search to convergence
        current_perm, current_ms = local_search(current_perm)
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm
            
        # Perturbation: Randomly reverse a segment of random length
        # Or move a block of random length to a random position
        # We use a mix.
        if random.random() < 0.5:
            # Reverse segment
            l = random.randint(0, n_jobs - 1)
            r = random.randint(l + 1, n_jobs - 1)
            # Ensure length is at least 2
            if r > l:
                perturbed = current_perm[:l] + current_perm[l:r+1][::-1] + current_perm[r+1:]
            else:
                perturbed = list(current_perm)
        else:
            # Or-opt style perturbation: move block of size 1-3
            block_size = random.randint(1, min(3, n_jobs))
            i = random.randint(0, n_jobs - block_size)
            block = current_perm[i:i+block_size]
            remaining = current_perm[:i] + current_perm[i+block_size:]
            j = random.randint(0, len(remaining))
            perturbed = remaining[:j] + block + remaining[j:]
            
        # Evaluate perturbed solution
        perturbed_ms = _calc_makespan_fast(processing_times, perturbed)
        
        # Accept if better, or with some probability if worse (to escape local minima)
        # Simple ILS accepts only if better or equal, but that can get stuck.
        # We use a simple acceptance rule: accept if better, or with prob 0.1 if worse.
        if perturbed_ms < current_ms or random.random() < 0.1:
            current_perm = perturbed
            current_ms = perturbed_ms
            
    return best_perm
# EVOLVE-BLOCK-END
