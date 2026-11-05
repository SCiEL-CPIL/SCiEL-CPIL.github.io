import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return -1

    best_index = -1
    best_machine = None
    max_duration = -1
    min_remaining_ops = float('inf')
    min_job_id = float('inf')

    for i, op in enumerate(ready_operations):
        options = op.get('machine_options', {})
        if not options:
            continue
            
        # Find the machine with the minimum duration for this operation
        # This assumes we can choose the machine to minimize the operation's completion time
        min_dur = float('inf')
        chosen_machine = None
        for m, d in options.items():
            if d < min_dur:
                min_dur = d
                chosen_machine = m
        
        # Heuristic: Longest Processing Time (LPT)
        # We want to schedule the operation with the largest processing time first.
        # This helps reduce the makespan by tackling bottlenecks early.
        
        # Tie-breaking strategy:
        # 1. Larger duration
        # 2. If duration is tied, prefer job with fewer remaining operations (to clear jobs)
        # 3. If still tied, prefer smaller job ID (determinism)
        
        rem_ops = op.get('remaining_ops', 0)
        job_id = op.get('job', i) # Use index as fallback for job ID
        
        if (min_dur > max_duration or 
            (min_dur == max_duration and rem_ops < min_remaining_ops) or
            (min_dur == max_duration and rem_ops == min_remaining_ops and job_id < min_job_id)):
            
            max_duration = min_dur
            best_index = i
            best_machine = chosen_machine
            min_remaining_ops = rem_ops
            min_job_id = job_id

    if best_index == -1:
        return -1

    # Return tuple (index, machine) to explicitly select the fastest machine
    return (best_index, best_machine)
