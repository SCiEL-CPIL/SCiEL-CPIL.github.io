"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]


def _compute_completion_array(processing_times, perm):
    """Compute full completion array for a permutation. Returns list of length n_machines."""
    n_machines = processing_times.shape[1]
    completion = [0.0] * n_machines
    
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion


def _evaluate_insertion_move(processing_times, seq, i, j, completion_prefix):
    """
    Evaluate the makespan after removing job at position i and inserting at position j.
    completion_prefix[k] = completion time on machine k after processing seq[0..k-1].
    We only need to recompute from min(i,j) to max(i,j) (adjusted).
    
    Returns the new makespan.
    """
    n_machines = processing_times.shape[1]
    n = len(seq)
    
    job = seq[i]
    
    # Determine the range of positions that change
    if j > i:
        # Remove at i, insert at j (which becomes j-1 after removal)
        # Positions i through j-1 in new seq are: old seq[i+1..j-1] then job
        # The prefix up to position i is unchanged
        start_pos = i
        # After removal, positions i..j-2 shift left; job goes at position j-1
        # New segment: seq[i+1], seq[i+2], ..., seq[j-1], job
        # Length of changed segment: (j-1) - i + 1 = j - i
        end_pos = j - 1  # position of job in new sequence
    else:
        # j <= i: remove at i, insert at j
        # Positions j through i-1 in new seq are: job, seq[j], seq[j+1], ..., seq[i-1]
        # The prefix up to position j is unchanged
        start_pos = j
        # New segment: job, seq[j], ..., seq[i-1]
        # Length: i - j + 1
        end_pos = i  # position of last shifted job in new sequence
    
    # Get the starting completion state
    if start_pos == 0:
        comp = [0.0] * n_machines
    else:
        comp = list(completion_prefix[start_pos])
    
    # Process the changed segment
    if j > i:
        # Changed positions in new sequence: i, i+1, ..., j-1
        # Jobs at these positions: seq[i+1], seq[i+2], ..., seq[j-1], seq[i] (the removed job)
        for pos in range(start_pos, end_pos + 1):
            if pos < end_pos:
                # Position pos in new seq corresponds to old position pos+1 (shifted left)
                row = processing_times[seq[pos + 1]]
            else:
                # Last position is the inserted job
                row = processing_times[job]
            
            for m in range(n_machines):
                if m == 0:
                    comp[0] += row[m]
                else:
                    c_prev = comp[m-1]
                    c_curr = comp[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    comp[m] = c_curr + row[m]
        
        # Now process the remaining unchanged suffix: positions end_pos+1 to n-1
        # These are old positions end_pos+1 to n-1 (i.e., seq[end_pos+1:] unchanged)
        for pos in range(end_pos + 1, n):
            row = processing_times[seq[pos]]
            for m in range(n_machines):
                if m == 0:
                    comp[0] += row[m]
                else:
                    c_prev = comp[m-1]
                    c_curr = comp[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    comp[m] = c_curr + row[m]
    else:
        # j <= i
        # Changed positions in new sequence: j, j+1, ..., i
        # Jobs: seq[i] (inserted), seq[j], seq[j+1], ..., seq[i-1]
        for pos in range(start_pos, end_pos + 1):
            if pos == start_pos:
                # First position is the inserted job
                row = processing_times[job]
            else:
                # Position pos in new seq corresponds to old position pos-1 (shifted right)
                row = processing_times[seq[pos - 1]]
            
            for m in range(n_machines):
                if m == 0:
                    comp[0] += row[m]
                else:
                    c_prev = comp[m-1]
                    c_curr = comp[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    comp[m] = c_curr + row[m]
        
        # Now process the remaining unchanged suffix: positions end_pos+1 to n-1
        # These are old positions end_pos+1 to n-1 (unchanged since we only shifted within j..i)
        for pos in range(end_pos + 1, n):
            row = processing_times[seq[pos]]
            for m in range(n_machines):
                if m == 0:
                    comp[0] += row[m]
                else:
                    c_prev = comp[m-1]
                    c_curr = comp[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    comp[m] = c_curr + row[m]
    
    return comp[-1]


def _compute_all_prefix_completions(processing_times, seq):
    """
    Compute completion array after each prefix of seq.
    Returns list of lists: prefix_comps[k] = completion after processing seq[0..k-1].
    prefix_comps[0] = [0, 0, ..., 0]
    prefix_comps[n] = completion after all jobs (makespan is prefix_comps[n][-1])
    """
    n_machines = processing_times.shape[1]
    n = len(seq)
    prefix_comps = [[0.0] * n_machines]
    
    comp = [0.0] * n_machines
    for k in range(n):
        row = processing_times[seq[k]]
        for m in range(n_machines):
            if m == 0:
                comp[0] += row[m]
            else:
                c_prev = comp[m-1]
                c_curr = comp[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                comp[m] = c_curr + row[m]
        prefix_comps.append(list(comp))
    
    return prefix_comps


def _neh_heuristic(processing_times):
    """
    NEH heuristic with incremental evaluation for insertion.
    """
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            # Build candidate sequence
            if pos == 0:
                cand = [job] + sequence
            elif pos == len(sequence):
                cand = sequence + [job]
            else:
                cand = sequence[:pos] + [job] + sequence[pos:]
            
            ms = _calc_makespan(processing_times, cand)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _fast_local_search(processing_times, sequence, deadline):
    """
    Fast insertion local search using precomputed prefix completions.
    For each move, only recompute the affected segment.
    """
    n_jobs = len(sequence)
    n_machines = processing_times.shape[1]
    current_seq = list(sequence)
    
    # Precompute prefix completions
    prefix_comps = _compute_all_prefix_completions(processing_times, current_seq)
    current_makespan = prefix_comps[-1][-1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                # Evaluate the move using incremental computation
                # We need prefix_comps[i] as the starting state
                new_makespan = _evaluate_insertion_move(
                    processing_times, current_seq, i, j, prefix_comps
                )
                
                if new_makespan < current_makespan:
                    # Accept the move
                    if j > i:
                        current_seq.pop(i)
                        current_seq.insert(j - 1, job)
                    else:
                        current_seq.pop(i)
                        current_seq.insert(j, job)
                    
                    current_makespan = new_makespan
                    improved = True
                    # Recompute prefix completions after accepting a move
                    prefix_comps = _compute_all_prefix_completions(processing_times, current_seq)
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing to escape local minima.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if n_jobs < 2:
            break
        i, j = random.sample(range(n_jobs), 2)
        new_seq = current_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
        if T < 1e-6:
            break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Fast local search with time limit
    best_seq, best_makespan = _fast_local_search(processing_times, initial_seq, deadline)
    
    # Step 3: Simulated Annealing if time permits
    if time.time() < deadline - 5.0:
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Step 4: Final fast local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _fast_local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
    
    return best_seq


# EVOLVE-BLOCK-END
