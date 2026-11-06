"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation efficiently using NumPy."""
    n_machines = processing_times.shape[1]
    # perm is an array of job indices
    # We need to compute the completion time on the last machine
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    # We can do this iteratively over machines or jobs.
    # Let's use a vectorized approach where possible, but the dependency is sequential.
    # A loop over jobs is O(n*m), which is fine.
    
    # To speed up, we can pre-extract the processing times for the permutation
    # P_perm[i, j] = processing_times[perm[i], j]
    P = processing_times[perm]
    
    # completion[m] stores the completion time of the current job on machine m
    completion = np.zeros(n_machines)
    
    for i in range(len(perm)):
        # For each job, compute completion on each machine
        # completion[m] = max(completion[m], completion[m-1]) + P[i, m]
        # This is a scan operation.
        # We can do it in a loop over machines, which is small.
        for m in range(n_machines):
            if m == 0:
                completion[0] += P[i, m]
            else:
                completion[m] = max(completion[m], completion[m-1]) + P[i, m]
    
    return completion[-1]


def _neh_heuristic(processing_times):
    """
    NEH heuristic: 
    1. Sort jobs by total processing time descending.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with the first job
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            # Calculate makespan for this new sequence
            # Optimization: we only need to check from pos onwards? 
            # Actually, the completion times before pos are unchanged, 
            # but the job at pos shifts. So we need to recompute from pos.
            # For simplicity and correctness, let's just compute the full makespan.
            # But that's O(n*m) per insertion, and O(n) insertions, so O(n^2*m).
            # For n=100, m=10, that's 10000 * 10 = 100k operations, which is fast.
            
            ms = _calc_makespan(processing_times, np.array(new_seq))
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    2-opt local search: try swapping adjacent jobs or moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, np.array(current_seq))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs of positions (i, j) with i < j
        # Swap and check if makespan improves
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n_jobs):
                if time.time() >= deadline:
                    break
                # Swap i and j
                current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
                new_makespan = _calc_makespan(processing_times, np.array(current_seq))
                if new_makespan < current_makespan:
                    current_makespan = new_makespan
                    improved = True
                else:
                    # Revert
                    current_seq[i], current_seq[j] = current_seq[j], current_seq[i]
    
    return current_seq, current_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # If the instance is small, we can do more exhaustive search
    # For now, use NEH + local search
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Local search with time limit
    best_seq, best_makespan = _local_search(processing_times, initial_seq, deadline)
    
    return best_seq


# EVOLVE-BLOCK-END
