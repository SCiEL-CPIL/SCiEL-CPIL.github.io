"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order in O(n*m)."""
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    machine_completion = [0.0] * n_machines
    for job_idx in order:
        prev = 0.0
        for m in range(n_machines):
            mc = machine_completion[m]
            if mc > prev:
                prev = mc
            prev += processing_times[job_idx, m]
            machine_completion[m] = prev
    return machine_completion[-1]


def _makespan_array(processing_times, order):
    """Vectorized-ish makespan using numpy for speed on larger problems."""
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    order_arr = np.asarray(order)
    job_times = processing_times[order_arr]  # (n_jobs, n_machines)
    # C[i][m] = max(C[i-1][m], C[i][m-1]) + P[i][m]
    # We can compute row by row
    n_jobs = len(order)
    # machine_completion tracks C[i-1][m]
    machine_completion = np.zeros(n_machines, dtype=float)
    for i in range(n_jobs):
        # C[i][m] = max(machine_completion[m], C[i][m-1]) + P[i][m]
        # C[i][m-1] for m=0 is 0
        # We need to do this sequentially per machine for the row dependency
        prev = 0.0
        for m in range(n_machines):
            mc = machine_completion[m]
            if mc > prev:
                prev = mc
            prev += job_times[i, m]
            machine_completion[m] = prev
    return machine_completion[-1]


def _neh_order(processing_times, rng=None):
    """NEH heuristic with best insertion."""
    if rng is None:
        rng = random.Random(42)
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(sequence) + 1):
            trial = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _random_order(n_jobs, rng):
    order = list(range(n_jobs))
    rng.shuffle(order)
    return order


def _sa_search(processing_times, initial_seq, time_limit, rng, tabu_size=20):
    """Simulated annealing with tabu list."""
    n = len(initial_seq)
    if n <= 1:
        return list(initial_seq)
    
    current_seq = list(initial_seq)
    current_ms = _makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Adaptive temperature
    T = current_ms * 0.1
    T_min = current_ms * 0.001
    cooling = 0.999
    
    # Tabu list: store (i, j, direction) for swaps, with tenure
    tabu = []
    tabu_tenure = max(5, n // 2)
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Choose neighborhood move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Swap
            i, j = rng.sample(range(n), 2)
            trial = current_seq.copy()
            trial[i], trial[j] = trial[j], trial[i]
            trial_ms = _makespan(processing_times, trial)
            key = (min(i,j), max(i,j), 'swap')
        elif move_type < 0.7:
            # Segment reversal
            if n < 2:
                break
            i, j = sorted(rng.sample(range(n), 2))
            trial = current_seq[:i] + current_seq[i:j+1][::-1] + current_seq[j+1:]
            trial_ms = _makespan(processing_times, trial)
            key = (i, j, 'rev')
        else:
            # Remove and reinsert
            i = rng.randint(0, n-1)
            j = rng.randint(0, n-1)
            if i == j:
                continue
            job = current_seq[i]
            trial = current_seq[:i] + current_seq[i+1:]
            # Insert at position j (adjust if j > i)
            insert_pos = j if j < i else j
            trial = trial[:insert_pos] + [job] + trial[insert_pos:]
            trial_ms = _makespan(processing_times, trial)
            key = (i, j, 'ins')
        
        delta = trial_ms - current_ms
        
        # Check tabu
        is_tabu = False
        for t_key, t_iter in tabu:
            if t_key == key and iteration - t_iter < tabu_tenure:
                is_tabu = True
                break
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        elif is_tabu:
            # Aspiration: accept if it improves best
            accept = trial_ms < best_ms
        else:
            # Simulated annealing acceptance
            if T > T_min:
                accept_prob = np.exp(-delta / T)
                accept = rng.random() < accept_prob
            else:
                accept = False
        
        if accept:
            current_seq = trial
            current_ms = trial_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
        
        # Update tabu
        tabu.append((key, iteration))
        if len(tabu) > tabu_tenure * 2:
            tabu = tabu[-tabu_tenure:]
        
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_seq


def _local_search_insertion(processing_times, sequence, max_iters=200, time_limit=2.0):
    """Insertion-based local search: try moving each job to every position."""
    n = len(sequence)
    if n <= 2:
        return list(sequence)
    
    current_seq = list(sequence)
    current_ms = _makespan(processing_times, current_seq)
    best_seq = list(current_seq)
    best_ms = current_ms
    
    start_time = time.time()
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            job = current_seq[i]
            remaining = current_seq[:i] + current_seq[i+1:]
            for pos in range(len(remaining) + 1):
                if pos == i or pos == i + 1:
                    continue
                trial = remaining[:pos] + [job] + remaining[pos:]
                trial_ms = _makespan(processing_times, trial)
                if trial_ms < current_ms - 1e-9:
                    current_seq = trial
                    current_ms = trial_ms
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_seq = list(current_seq)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_seq


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using
    NEH + Simulated Annealing + Tabu + Insertion Local Search.
    """
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin
    rng = random.Random(12345)
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # 1. NEH
    neh_seq = _neh_order(processing_times, rng)
    initial_solutions.append(neh_seq)
    
    # 2. Random
    if n_jobs <= 50:
        initial_solutions.append(_random_order(n_jobs, rng))
    
    # 3. Perturbed NEH (swap two random)
    perturbed = list(neh_seq)
    if n_jobs >= 2:
        i, j = rng.sample(range(n_jobs), 2)
        perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        initial_solutions.append(perturbed)
    
    # 4. Another perturbed NEH
    perturbed2 = list(neh_seq)
    if n_jobs >= 3:
        i, j, k = rng.sample(range(n_jobs), 3)
        perturbed2[i], perturbed2[j] = perturbed2[j], perturbed2[i]
        perturbed2[j], perturbed2[k] = perturbed2[k], perturbed2[j]
        initial_solutions.append(perturbed2)
    
    # Evaluate all initial solutions
    best_seq = initial_solutions[0]
    best_ms = _makespan(processing_times, best_seq)
    for sol in initial_solutions[1:]:
        ms = _makespan(processing_times, sol)
        if ms < best_ms:
            best_ms = ms
            best_seq = sol
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 0.5:
        return best_seq
    
    # Phase 1: SA on the best initial solution
    sa_time = remaining * 0.4
    if sa_time > 0.5:
        sa_result = _sa_search(processing_times, best_seq, sa_time, rng)
        sa_ms = _makespan(processing_times, sa_result)
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_seq = sa_result
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 0.5:
        return best_seq
    
    # Phase 2: Insertion local search on SA result
    ls_time = min(remaining * 0.3, 5.0)
    if ls_time > 0.3:
        ls_result = _local_search_insertion(processing_times, best_seq, 
                                             max_iters=500, time_limit=ls_time)
        ls_ms = _makespan(processing_times, ls_result)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_seq = ls_result
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 1.0:
        return best_seq
    
    # Phase 3: SA on other initial solutions (if time permits)
    for sol in initial_solutions[1:]:
        if time.time() - start_time > deadline - 1.0:
            break
        sol_ms = _makespan(processing_times, sol)
        # Only if this solution is competitive
        if sol_ms < best_ms * 1.1:
            sa_time2 = min(remaining * 0.2, 3.0)
            if sa_time2 > 0.3:
                sa_result2 = _sa_search(processing_times, sol, sa_time2, rng)
                sa_ms2 = _makespan(processing_times, sa_result2)
                if sa_ms2 < best_ms:
                    best_ms = sa_ms2
                    best_seq = sa_result2
                elapsed = time.time() - start_time
                remaining = deadline - elapsed
                if remaining < 1.0:
                    break
    
    return best_seq
# EVOLVE-BLOCK-END
