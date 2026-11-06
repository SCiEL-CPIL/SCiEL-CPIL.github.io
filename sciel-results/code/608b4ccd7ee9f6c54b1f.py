"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation."""
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    if n_machines == 1:
        return float(processing_times[list(perm), 0].sum())
    completion = [0.0] * n_machines
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m - 1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    return completion[-1]


def _calc_makespan_fast(processing_times, perm):
    """Vectorized makespan calculation using numpy."""
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return float(processing_times[list(perm), 0].sum())
    
    pt = processing_times[np.array(perm)]
    n_jobs = len(perm)
    
    C = np.empty((n_jobs, n_machines), dtype=np.float64)
    C[:, 0] = np.cumsum(pt[:, 0])
    
    for m in range(1, n_machines):
        prev_col = np.empty(n_jobs, dtype=np.float64)
        prev_col[0] = C[0, m - 1]
        prev_col[1:] = C[:-1, m]
        C[:, m] = np.maximum(C[:, m - 1], prev_col) + pt[:, m]
    
    return float(C[-1, -1])


def _insertion_delta(processing_times, seq, i, j):
    """
    Compute the makespan change when moving job at position i to position j.
    Returns the delta (new_makespan - old_makespan).
    """
    n_jobs = len(seq)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return 0.0
    
    # Compute completion times for original sequence
    C_orig = np.empty((n_jobs, n_machines), dtype=np.float64)
    pt_orig = processing_times[np.array(seq)]
    C_orig[:, 0] = np.cumsum(pt_orig[:, 0])
    for m in range(1, n_machines):
        prev_col = np.empty(n_jobs, dtype=np.float64)
        prev_col[0] = C_orig[0, m - 1]
        prev_col[1:] = C_orig[:-1, m]
        C_orig[:, m] = np.maximum(C_orig[:, m - 1], prev_col) + pt_orig[:, m]
    
    old_makespan = C_orig[-1, -1]
    
    # Build new sequence
    if j > i:
        new_seq = list(seq[:i]) + list(seq[i+1:j]) + [seq[i]] + list(seq[j:])
    else:
        new_seq = list(seq[:j]) + [seq[i]] + list(seq[j:i]) + list(seq[i+1:])
    
    # Compute makespan for new sequence
    C_new = np.empty((n_jobs, n_machines), dtype=np.float64)
    pt_new = processing_times[np.array(new_seq)]
    C_new[:, 0] = np.cumsum(pt_new[:, 0])
    for m in range(1, n_machines):
        prev_col = np.empty(n_jobs, dtype=np.float64)
        prev_col[0] = C_new[0, m - 1]
        prev_col[1:] = C_new[:-1, m]
        C_new[:, m] = np.maximum(C_new[:, m - 1], prev_col) + pt_new[:, m]
    
    new_makespan = C_new[-1, -1]
    return new_makespan - old_makespan


def _neh_heuristic(processing_times):
    """
    NEH heuristic: Sort jobs by total processing time descending,
    then insert each job into the position that minimizes makespan.
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
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _random_permutation(n_jobs):
    """Generate a random permutation of job indices."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _reverse_permutation(sequence):
    """Return the reversed sequence."""
    return list(reversed(sequence))


def _shifted_permutation(sequence, shift):
    """Return the sequence rotated by shift positions."""
    n = len(sequence)
    if shift % n == 0:
        return list(sequence)
    return list(sequence[shift % n:]) + list(sequence[:shift % n])


def _swap_pair(sequence, i, j):
    """Return a new sequence with elements at i and j swapped."""
    new_seq = list(sequence)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def _insertion_move(sequence, i, j):
    """
    Remove element at position i and insert it at position j.
    If j > i, adjust for the shift caused by removal.
    """
    new_seq = list(sequence)
    job = new_seq.pop(i)
    if j > i:
        insert_idx = j - 1
    else:
        insert_idx = j
    new_seq.insert(insert_idx, job)
    return new_seq


def _generate_candidate(processing_times, n_jobs, strategy, rng_seed=None):
    """
    Generate a single candidate permutation based on the given strategy.
    strategy: 'neh', 'random', 'reversed_neh', 'shifted_neh', 'random_swap_neh'
    """
    if strategy == 'neh':
        return _neh_heuristic(processing_times)
    elif strategy == 'random':
        return _random_permutation(n_jobs)
    elif strategy == 'reversed_neh':
        return _reverse_permutation(_neh_heuristic(processing_times))
    elif strategy == 'shifted_neh':
        base = _neh_heuristic(processing_times)
        shift = random.randint(1, max(1, n_jobs - 1))
        return _shifted_permutation(base, shift)
    elif strategy == 'random_swap_neh':
        base = _neh_heuristic(processing_times)
        if n_jobs >= 2:
            i, j = random.sample(range(n_jobs), 2)
            return _swap_pair(base, i, j)
        return base
    else:
        return _random_permutation(n_jobs)


def _local_search(processing_times, sequence, deadline):
    """
    Best-improvement insertion local search with vectorized evaluation.
    Tries all insertion moves and picks the best improvement per pass.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    while time.time() < deadline:
        best_delta = 0.0
        best_i = -1
        best_j = -1
        
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            for j in range(n_jobs):
                if j == i:
                    continue
                
                # Build new sequence for evaluation
                if j > i:
                    new_seq = list(current_seq[:i]) + list(current_seq[i+1:j]) + [current_seq[i]] + list(current_seq[j:])
                else:
                    new_seq = list(current_seq[:j]) + [current_seq[i]] + list(current_seq[j:i]) + list(current_seq[i+1:])
                
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                delta = new_makespan - current_makespan
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i < 0:
            break  # No improvement found
        
        # Apply the best move
        i, j = best_i, best_j
        if j > i:
            current_seq = list(current_seq[:i]) + list(current_seq[i+1:j]) + [current_seq[i]] + list(current_seq[j:])
        else:
            current_seq = list(current_seq[:j]) + [current_seq[i]] + list(current_seq[j:i]) + list(current_seq[i+1:])
        current_makespan += best_delta
    
    return current_seq, current_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing with insertion perturbations and adaptive cooling.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.2
    min_T = current_makespan * 0.001
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and T > min_T:
        # Use insertion moves (more effective than swaps for flow shops)
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
        
        if j > i:
            new_seq = list(current_seq[:i]) + list(current_seq[i+1:j]) + [current_seq[i]] + list(current_seq[j:])
        else:
            new_seq = list(current_seq[:j]) + [current_seq[i]] + list(current_seq[j:i]) + list(current_seq[i+1:])
        
        new_makespan = _calc_makespan_fast(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart from best with higher temperature
                current_seq = list(best_seq)
                current_makespan = best_makespan
                T = best_makespan * 0.1
                no_improve_count = 0
                max_no_improve = int(max_no_improve * 1.2)
        
        T *= cooling_rate
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate diverse initial candidates
    strategies = [
        'neh',
        'random',
        'reversed_neh',
        'shifted_neh',
        'random_swap_neh',
    ]
    
    # Add more random variants for diversity
    for _ in range(5):
        strategies.append('random')
    
    candidates = []
    for strat in strategies:
        if time.time() >= deadline:
            break
        cand = _generate_candidate(processing_times, n_jobs, strat)
        candidates.append(cand)
    
    # Evaluate all candidates and keep the best
    best_seq = None
    best_makespan = float('inf')
    
    for cand in candidates:
        if time.time() >= deadline:
            break
        ms = _calc_makespan_fast(processing_times, cand)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = list(cand)
    
    if best_seq is None:
        best_seq = list(range(n_jobs))
        best_makespan = _calc_makespan_fast(processing_times, best_seq)
    
    # Phase 1: Local search on the best candidate
    if time.time() < deadline - 10.0:
        ls_deadline = min(deadline - 10.0, time.time() + 40.0)
        ls_seq, ls_makespan = _local_search(processing_times, best_seq, ls_deadline)
        if ls_makespan < best_makespan:
            best_seq = ls_seq
            best_makespan = ls_makespan
    
    # Phase 2: Simulated annealing if time permits
    if time.time() < deadline - 15.0:
        sa_deadline = min(deadline - 5.0, time.time() + 50.0)
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, sa_deadline)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Phase 3: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
            best_makespan = final_makespan
    
    return best_seq


# EVOLVE-BLOCK-END
