"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a greedy insertion heuristic.
    At each step, select the job that results in the minimum incremental makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # If the problem is very large, fall back to a faster heuristic to ensure completion
    # O(n^2 * m) is generally fine for n < 1000, but let's be safe.
    if n_jobs * n_machines > 50000:
        # Fast fallback: Longest Processing Time (LPT)
        return list(np.argsort(-processing_times.sum(axis=1)))

    # Greedy construction:
    # We maintain the current sequence and the completion times on each machine.
    # For each new job, we calculate the resulting makespan if added to the end.
    # We pick the job that minimizes the new makespan.
    
    remaining_jobs = list(range(n_jobs))
    current_sequence = []
    
    # completion_times[i] is the time when machine i finishes its last job in current_sequence
    completion_times = np.zeros(n_machines)
    
    while remaining_jobs:
        best_job = -1
        best_makespan = float('inf')
        
        # Check time limit
        if time.time() > deadline:
            # Return current best partial solution padded with remaining jobs
            # This is a rare edge case, but ensures we return a valid permutation
            current_sequence.extend(remaining_jobs)
            break
            
        for job in remaining_jobs:
            # Calculate makespan if we append 'job' to current_sequence
            # The start time on machine 0 is completion_times[0]
            # The start time on machine k is max(completion_times[k-1], completion_times[k])
            
            # We can simulate this quickly
            # start_m0 = completion_times[0]
            # finish_m0 = start_m0 + p[job, 0]
            # start_m1 = max(finish_m0, completion_times[1])
            # finish_m1 = start_m1 + p[job, 1]
            # ...
            
            # To optimize, we can just compute the final makespan.
            # Since n_machines is usually small, a loop is fine.
            
            prev_finish = completion_times[0]
            curr_finish = prev_finish + processing_times[job, 0]
            
            for m in range(1, n_machines):
                prev_finish = curr_finish
                curr_finish = max(prev_finish, completion_times[m]) + processing_times[job, m]
                
            # The makespan is the finish time on the last machine
            new_makespan = curr_finish
            
            if new_makespan < best_makespan:
                best_makespan = new_makespan
                best_job = job
                
        if best_job == -1:
            # Should not happen if remaining_jobs is not empty
            break
            
        # Add best_job to the sequence
        current_sequence.append(best_job)
        remaining_jobs.remove(best_job)
        
        # Update completion_times for the next iteration
        # We need to re-simulate the full sequence to get accurate completion_times for all machines?
        # No, we can update incrementally.
        # However, the incremental update of completion_times requires knowing the start time on each machine.
        # Let's just re-calculate the completion times for the new sequence? 
        # That would be O(n^2 * m) total which is fine.
        # Or we can store the start times.
        
        # Actually, for the next step, we need `completion_times` to represent the state AFTER adding best_job.
        # Let's re-compute completion_times for the current_sequence to be safe and simple.
        # Wait, re-computing is O(n*m) per step, so total O(n^2*m).
        # Let's do it efficiently.
        
        # Recalculate completion_times for the new sequence
        completion_times = np.zeros(n_machines)
        for job in current_sequence:
            prev_finish = completion_times[0]
            curr_finish = prev_finish + processing_times[job, 0]
            completion_times[0] = curr_finish
            for m in range(1, n_machines):
                prev_finish = curr_finish
                curr_finish = max(prev_finish, completion_times[m]) + processing_times[job, m]
                completion_times[m] = curr_finish

    return current_sequence
# EVOLVE-BLOCK-END
