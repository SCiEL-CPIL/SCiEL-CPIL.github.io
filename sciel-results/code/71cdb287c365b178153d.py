"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # C[m] is the completion time of the current job on machine m
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Or-opt Local Search
    # We perform iterative improvement using Or-opt moves.
    # An Or-opt move takes a block of 1, 2, or 3 jobs and inserts it at a new position.
    # We do this until no improvement is found in a full pass or time runs out.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_order)
        
        # To avoid bias towards early positions, we can randomize the start or iterate systematically.
        # Systematic is safer for determinism but might get stuck in local minima faster.
        # Let's use a simple systematic pass first. If we want more exploration, we could randomize.
        # For now, systematic pass with block sizes 1, 2, 3.
        
        for block_size in [1, 2, 3]:
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    return best_order
                
                # Extract block
                block = current_order[i:i+block_size]
                
                # Remove block from current position
                # We need to evaluate insertion at all other valid positions
                # Positions are 0 to n - block_size
                # We skip the original position i (and i+1, etc if considering shifts, but standard is just index i)
                
                # To be efficient, we can construct the new list and evaluate.
                # Since n is likely small enough (<100s), O(n^2) evaluation per pass is acceptable if passes are few.
                
                best_pos = -1
                best_ms = current_makespan
                
                for j in range(n - block_size + 1):
                    if j == i:
                        continue
                    
                    # Construct new order
                    # Remove block from i
                    # Insert block at j
                    # Note: If j > i, the indices shift after removal.
                    # Easier way: create a list without the block, then insert.
                    
                    # Create list of remaining jobs
                    remaining = current_order[:i] + current_order[i+block_size:]
                    
                    # Insert block at j
                    # j is the index in the *new* list (length n - block_size)
                    # Valid j range for insertion is 0 to len(remaining)
                    # len(remaining) = n - block_size
                    # So j can be 0 to n - block_size
                    
                    new_order = remaining[:j] + block + remaining[j:]
                    
                    ms = _makespan(new_order, processing_times)
                    
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                        
                if best_pos != -1:
                    # Apply the best move found for this block_size and i
                    remaining = current_order[:i] + current_order[i+block_size:]
                    current_order = remaining[:best_pos] + block + remaining[best_pos:]
                    current_makespan = best_ms
                    
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = current_order[:]
                        
                    improved = True
                    # After a move, the indices change, so we should restart the loop for this block_size
                    # or at least re-evaluate. Restarting the outer `for i` loop is safer.
                    # However, breaking out of nested loops in Python is messy.
                    # Let's just break out and let the `while improved` loop handle it.
                    # We need to signal that we should restart the systematic scan.
                    # A simple restart is: set a flag and break.
                    break
            
            if improved:
                # If we made an improvement, break out of the block_size loop to restart the i-loop
                # with the new configuration
                break
                
    return best_order

# EVOLVE-BLOCK-END
