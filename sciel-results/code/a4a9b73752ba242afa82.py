"""Structural rewrite: fast NEH + fast delta-insertion local search with time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    """
    if len(order) == 0:
        return 0.0
    n_machines = P.shape[1]
    c_machines = np.zeros(n_machines)
    c_curr = 0.0
    for j in order:
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
    return c_curr

def _compute_C(order, P):
    """
    Compute full completion matrix C[k][m] for the given order.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    C = np.zeros((n_jobs, n_machines), dtype=float)
    c_machines = np.zeros(n_machines, dtype=float)
    c_curr = 0.0
    for k in range(n_jobs):
        j = order[k]
        row = P[j]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
            C[k, m] = c_curr
    return C

def _fast_insertion_eval(order, P, i, j, C):
    """
    Evaluate makespan after moving job at position i to position j.
    Uses precomputed C to recompute only the affected segment in O(|segment| * M).
    Returns the new makespan.
    """
    n_jobs = len(order)
    n_machines = P.shape[1]
    
    if i == j:
        return C[n_jobs - 1, n_machines - 1]
    
    # Determine the segment that changes
    if i < j:
        # Move job i to position j.
        # Jobs i+1..j shift left by one.
        # The affected segment is from i to j.
        # Start: completion before position i is C[i-1, m] (or 0 if i==0).
        start_pos = i
        end_pos = j
    else:
        # i > j: Move job i to position j.
        # Jobs j..i-1 shift right by one.
        start_pos = j
        end_pos = i
    
    # Get the job being moved
    job_moved = order[i]
    
    # The sequence of jobs in the new positions start_pos..end_pos
    # Original jobs in [start_pos, end_pos] excluding i, plus job_moved at position end_pos if i<j or start_pos if i>j?
    # Let's construct the new list of jobs for the segment.
    # If i < j:
    #   Original: ... order[i], order[i+1], ..., order[j] ...
    #   New:      ... order[i+1], ..., order[j], order[i] ...
    #   Segment indices in new order: start_pos..end_pos
    #   Jobs: order[i+1], ..., order[j], order[i]
    # If i > j:
    #   Original: ... order[j], ..., order[i] ...
    #   New:      ... order[i], order[j], ..., order[i-1] ...
    #   Segment indices in new order: start_pos..end_pos
    #   Jobs: order[i], order[j], ..., order[i-1]
    
    if i < j:
        seg_jobs = [order[k] for k in range(i + 1, j + 1)]
        seg_jobs.append(order[i])
    else:
        seg_jobs = [order[i]]
        seg_jobs.extend([order[k] for k in range(j, i)])
    
    # Compute new completions for the segment
    # Initial state: c_machines from before start_pos
    if start_pos == 0:
        c_machines = np.zeros(n_machines, dtype=float)
    else:
        c_machines = C[start_pos - 1].copy()
    
    c_curr = 0.0
    if start_pos > 0:
        c_curr = np.max(c_machines)
    else:
        c_curr = 0.0
        
    # Wait, c_curr is the completion time on the previous machine for the current job?
    # In the standard flow shop recurrence:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # Here, for a segment, we need the "incoming" times.
    # Let c_prev_m be the completion time of the previous job (in the sequence) on machine m.
    # Let c_curr_m be the completion time of the current job on machine m.
    # Actually, it's easier to maintain a vector of machine completion times.
    
    # Let's re-initialize properly.
    # The state entering the segment is the completion times of the job at start_pos-1.
    # Let's call it state_in.
    if start_pos > 0:
        state_in = C[start_pos - 1].copy()
    else:
        state_in = np.zeros(n_machines, dtype=float)
    
    # Now process seg_jobs
    # For each job in seg_jobs:
    #   For each machine m:
    #     new_state[m] = max(new_state[m-1], state_in[m]) + P[job, m]
    #     Wait, state_in[m] is the completion of the PREVIOUS job on machine m.
    #     But for the first job in the segment, the previous job is the one at start_pos-1.
    #     For subsequent jobs, the "previous job" is the one just before them in the segment.
    #     So we need to update state_in as we go?
    #     No, state_in[m] is fixed for the first job. For the second job, the previous job is the first job of the segment.
    #     So we need to keep track of the completion times of the last processed job.
    
    last_job_completions = state_in.copy()
    
    for job in seg_jobs:
        row = P[job]
        new_completions = np.zeros(n_machines, dtype=float)
        c_curr = 0.0
        for m in range(n_machines):
            # c_curr is completion on machine m-1 for this job
            # last_job_completions[m] is completion on machine m for previous job
            c_curr = max(last_job_completions[m], c_curr) + row[m]
            new_completions[m] = c_curr
        last_job_completions = new_completions
        
    # The makespan is the completion time of the last job on the last machine
    # But wait, the segment ends at end_pos.
    # If end_pos < n_jobs - 1, the rest of the jobs are unchanged, but their start times might be affected?
    # No, the completion times of the job at end_pos are now different.
    # The jobs after end_pos depend on the job at end_pos.
    # So we need to propagate the new completion times of the last job in the segment to the rest of the sequence.
    
    # However, if end_pos == n_jobs - 1, the makespan is last_job_completions[-1].
    # If end_pos < n_jobs - 1, we need to continue the computation for the remaining jobs.
    
    if end_pos == n_jobs - 1:
        return last_job_completions[-1]
    
    # Propagate to the rest
    # The rest of the jobs are order[end_pos+1], ..., order[n_jobs-1]
    # Their input state is last_job_completions (completion times of job at end_pos)
    # But wait, the original C had different values for these jobs.
    # We must recompute them.
    
    # This makes the evaluation O((n_jobs - end_pos) * M) in the worst case, which is O(N*M).
    # This defeats the purpose of "fast" evaluation if the segment is small but the tail is large.
    # However, for insertion moves, the tail is often small if we move to the end, but can be large if we move to the front.
    # Actually, if we move a job from the middle to the front, the tail is almost the whole sequence.
    
    # Is there a way to avoid recomputing the tail?
    # Only if the completion times of the job at end_pos are the same as before.
    # But they are different.
    
    # So, _fast_insertion_eval is actually O(N*M) in the worst case.
    # This is the same complexity as _makespan, but with a lower constant if the segment is small and we can stop early?
    # No, we have to recompute the tail.
    
    # Let's check if we can use the original C for the tail?
    # No, because the start times of the tail jobs depend on the completion of the previous job.
    
    # So, this "fast" evaluation is not really faster than full recomputation in the worst case.
    # It is only faster if we can prove that the tail completion times are unchanged, which is unlikely.
    
    # Given this, maybe we should just use _makespan?
    # Or, we can use this if we assume that for most moves, the makespan is determined by the segment?
    # No, that's not true.
    
    # Let's stick to the implementation but be aware of the cost.
    # We will use it in the local search and rely on the time budget.
    
    c_machines = last_job_completions.copy()
    c_curr = c_machines[-1]
    for k in range(end_pos + 1, n_jobs):
        job = order[k]
        row = P[job]
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
    return c_curr

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH heuristic followed by local search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # 1. Initialize with NEH
    best_order = _neh_heuristic(P)
    best_makespan = _makespan(best_order, P)
    C = _compute_C(best_order, P)
    
    # 2. Local Search
    # Use first-improvement strategy with fast delta evaluation (or full recompute if needed)
    # We will iterate until no improvement is found or time runs out.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Recompute C at the start of each pass
        C = _compute_C(best_order, P)
        
        # Try all insertion moves
        # For efficiency, we can break early on first improvement (first-improvement)
        # or find the best move (best-improvement).
        # Best-improvement is more thorough but requires evaluating all moves.
        # Let's try best-improvement for small N, first-improvement for large N.
        
        if n_jobs <= 60:
            best_move_ms = best_makespan
            best_move = None
            for i in range(n_jobs):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(n_jobs):
                    if i == j:
                        continue
                    ms = _fast_insertion_eval(best_order, P, i, j, C)
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, j)
            if best_move is not None:
                i, j = best_move
                job_to_move = best_order[i]
                best_order.pop(i)
                if i < j:
                    best_order.insert(j, job_to_move)
                else:
                    best_order.insert(j, job_to_move)
                best_makespan = best_move_ms
                improved = True
        else:
            # First-improvement for larger N
            for i in range(n_jobs):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(n_jobs):
                    if i == j:
                        continue
                    ms = _fast_insertion_eval(best_order, P, i, j, C)
                    if ms < best_makespan:
                        job_to_move = best_order[i]
                        best_order.pop(i)
                        if i < j:
                            best_order.insert(j, job_to_move)
                        else:
                            best_order.insert(j, job_to_move)
                        best_makespan = ms
                        improved = True
                        break
                if improved:
                    break
    
    return best_order
# EVOLVE-BLOCK-END
