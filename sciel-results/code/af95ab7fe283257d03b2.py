"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    
    for job in perm:
        # Access the row of processing times for this job
        row = processing_times[job]
        # Iterate over machines
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                # completion[m] = max(completion[m], completion[m-1]) + row[m]
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
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
    Insertion local search: try moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs of positions (i, j)
        # Move job at i to position j
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            # Remove job at i
            # We can simulate the move by creating a new list, but that's O(N).
            # Since N is small, it's fine. Or we can manipulate the list in place.
            # Let's just create a new list for simplicity, as N is likely < 1000.
            # Actually, for speed, we can try to avoid full list copy if possible, 
            # but _calc_makespan takes a list, so we pass the modified list.
            
            # We need to try inserting at all positions j != i
            # To avoid modifying the original list until we find an improvement,
            # we can create a temporary list or carefully modify and revert.
            
            # Efficient approach:
            # 1. Remove job from i
            # 2. Insert at j
            # 3. Check makespan
            # 4. Revert if not improved
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Perform move
                # If j > i, removing at i shifts j-1. Inserting at j in the new list 
                # is equivalent to inserting at j-1 in the original list? 
                # It's easier to just use list operations.
                
                seq_copy = current_seq[:]
                job_to_move = seq_copy.pop(i)
                # Adjust insertion index if j > i
                # If we pop i, elements after i shift left by 1.
                # So if we want to insert at position j in the original sequence,
                # and j > i, the target index in the popped list is j - 1.
                # If j < i, the target index is j.
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                seq_copy.insert(insert_idx, job_to_move)
                
                new_makespan = _calc_makespan(processing_times, seq_copy)
                
                if new_makespan < current_makespan:
                    current_seq = seq_copy
                    current_makespan = new_makespan
                    improved = True
                    # After an improvement, we could restart the i loop or just continue?
                    # Continuing is fine, but the indices might be slightly off relative to the new state.
                    # Restarting i loop is safer to ensure we explore all moves from the new state.
                    # However, breaking out of j loop and restarting i loop is standard.
                    break
            
            # If we broke out of j loop due to improvement, we should restart i loop?
            # The `improved` flag is set. The while loop will check it.
            # But we are still in the `for i` loop. We should break the `for i` loop to restart from i=0.
            if improved:
                break
    
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
