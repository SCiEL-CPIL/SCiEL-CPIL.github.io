"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_makespan(seq, pt):
    """
    Calculate the makespan for a given sequence of jobs.
    seq: list of job indices
    pt: np.ndarray of processing times [jobs, machines]
    """
    if not seq:
        return 0.0
    n_machines = pt.shape[1]
    # C[m] is the completion time of the last job on machine m
    # We can simulate the flow shop
    # For efficiency, we can track the completion time of the current job on each machine
    # Actually, a standard simulation is:
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    # We only need the final C[n-1][m-1]
    
    # Vectorized approach for small n_machines is tricky due to dependency.
    # Loop over jobs is usually fast enough if n_jobs is not huge.
    # Let's use a simple loop which is O(n_jobs * n_machines).
    
    # Initialize completion times for the first job
    # C_prev is completion times on each machine for the previous job
    # C_curr is completion times on each machine for the current job
    
    # To make it faster, we can use a running total for the first machine?
    # No, flow shop dependencies are complex.
    
    # Let's use the standard algorithm:
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + P[j][k]
    
    # We can maintain an array `C_mach` of size n_machines representing completion time of the last processed job on each machine.
    # But we also need the completion time of the previous job on the *same* machine?
    # Actually, for a specific job j, its completion time on machine k depends on:
    # 1. Completion of job j on machine k-1
    # 2. Completion of previous job j-1 on machine k
    
    # Let's simulate job by job.
    # c_prev_k: completion time of previous job on machine k
    # c_curr_k: completion time of current job on machine k
    
    # Initialize c_prev to 0
    c_prev = np.zeros(n_machines)
    c_curr = np.zeros(n_machines)
    
    for job in seq:
        # c_curr[0] = c_prev[0] + pt[job, 0]
        # c_curr[k] = max(c_prev[k], c_curr[k-1]) + pt[job, k]
        
        # This loop is small (n_machines), so it's fast.
        c_curr[0] = c_prev[0] + pt[job, 0]
        for k in range(1, n_machines):
            c_curr[k] = max(c_prev[k], c_curr[k-1]) + pt[job, k]
        
        # Swap c_prev and c_curr for next iteration
        c_prev, c_curr = c_curr, c_prev
        
    return c_prev[n_machines - 1]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. NEH Heuristic Construction
    # Sort jobs by decreasing total processing time
    total_times = processing_times.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    
    # Insert remaining jobs
    for job in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _calc_makespan(new_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, job)
    
    best_seq = seq
    best_ms = best_makespan
    
    # 2. Local Search with Insertion Moves
    # We try to move each job to every other position.
    # To save time, we can limit the number of full passes or check time frequently.
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        n = len(best_seq)
        
        # Iterate through all jobs
        for i in range(n):
            if time.time() > deadline:
                break
            job = best_seq[i]
            # Remove job from current position
            # We don't need to physically remove and re-insert for every trial, 
            # but for simplicity and correctness with the helper function, we can.
            # Optimization: We can compute the makespan change more efficiently, 
            # but _calc_makespan is O(n*m). If n*m is small, it's fine.
            
            for j in range(n):
                if i == j:
                    continue
                # Skip if j is adjacent to i? No, sometimes adjacent moves are useful.
                # Construct new sequence
                # Move job from i to j
                # If j > i, the indices shift.
                # Let's create a copy for safety and simplicity, or do it in-place carefully.
                # Given n is likely small (e.g., < 50), copy is fast.
                
                new_seq = best_seq.copy()
                # Remove at i
                new_seq.pop(i)
                # Adjust j if j > i
                target_j = j - 1 if j > i else j
                # Insert at target_j
                new_seq.insert(target_j, job)
                
                ms = _calc_makespan(new_seq, processing_times)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = new_seq
                    # Update i and j to reflect the new position?
                    # The job is now at target_j.
                    # We can break inner loop to restart? Or just continue.
                    # Restarting is safer to avoid missing improvements caused by the move.
                    # But for speed, we can just mark improved and continue.
                    # Actually, if we move a job, the indices of other jobs change.
                    # It's better to break the inner loop and restart the outer loop or adjust.
                    # Let's break inner and let the while loop handle it?
                    # No, we are in a while loop. Let's break inner, set improved=True, 
                    # and continue outer? 
                    # If we continue outer, we might use stale indices for subsequent i.
                    # Safest: Break inner, set improved=True, and break outer to restart the pass?
                    # That might be slow.
                    # Alternative: Just update best_seq and best_ms, and continue.
                    # The next iteration of j will use the updated best_seq? 
                    # No, best_seq is updated, but the loop variable j is based on the old n.
                    # This is dangerous.
                    
                    # Let's use a simpler strategy: 
                    # If an improvement is found, update best_seq and best_ms, 
                    # and restart the local search pass (break both loops).
                    # This is "First Improvement" strategy, which is often faster than "Best Improvement".
                    improved = True
                    break
            if improved:
                break
        
        # If no improvement in a full pass, stop.
    
    return best_seq

# EVOLVE-BLOCK-END
