"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys


def _makespan_vectorized(processing_times, order):
    """Compute makespan using vectorized numpy operations."""
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    if len(order) == 1:
        return float(processing_times[order[0]].sum())
    
    # pt: (n_jobs, n_machines) submatrix in order
    pt = processing_times[order]
    
    # C[j, m] = completion time of job j on machine m
    # C[j, 0] = C[j-1, 0] + pt[j, 0]
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[j, m]
    
    n_jobs = len(order)
    # We can compute column by column
    C_prev = np.zeros(n_machines)
    for j in range(n_jobs):
        C_curr = np.empty(n_machines)
        C_curr[0] = C_prev[0] + pt[j, 0]
        if n_machines > 1:
            # C_curr[m] = max(C_prev[m], C_curr[m-1]) + pt[j, m]
            # This is a sequential dependency along m, but we can do it in a loop
            for m in range(1, n_machines):
                C_curr[m] = max(C_prev[m], C_curr[m-1]) + pt[j, m]
        C_prev = C_curr
    return float(C_prev[-1])


def _makespan(processing_times, order):
    """Compute makespan for a given permutation of job indices."""
    return _makespan_vectorized(processing_times, order)


def _neh(processing_times):
    """NEH heuristic: order by decreasing total processing time, insert each job in best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _neh_variants(processing_times):
    """Multiple NEH variants with different sorting criteria."""
    n_jobs, n_machines = processing_times.shape
    
    # Variant 1: total processing time
    totals = processing_times.sum(axis=1)
    order1 = list(np.argsort(-totals))
    
    # Variant 2: average processing time
    avg = processing_times.mean(axis=1)
    order2 = list(np.argsort(-avg))
    
    # Variant 3: first machine processing time
    order3 = list(np.argsort(-processing_times[:, 0]))
    
    # Variant 4: last machine processing time
    order4 = list(np.argsort(-processing_times[:, -1]))
    
    # Variant 5: Palmer
    priorities = np.zeros(n_jobs)
    for j in range(n_jobs):
        priorities[j] = (n_jobs + 1 - j) * processing_times[j, 0] - j * processing_times[j, n_machines - 1]
    order5 = list(np.argsort(-priorities))
    
    results = []
    for init_order in [order1, order2, order3, order4, order5]:
        # Run NEH insertion
        seq = [init_order[0]]
        for job in init_order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = _makespan(processing_times, trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        results.append(seq)
    
    return results


def _local_search(processing_times, order, time_limit=10.0):
    """Efficient local search with adjacent swaps, segment reversals, and insert moves."""
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    # Pre-extract processing times for the current order to speed up evaluations
    pt = processing_times  # (n_jobs, n_machines)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Pass 1: Adjacent swaps
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            # Swap positions i and i+1
            trial = best_order[:]
            trial[i], trial[i+1] = trial[i+1], trial[i]
            ms = _makespan(processing_times, trial)
            if ms < best_ms - 1e-10:
                best_order = trial
                best_ms = ms
                improved = True
                # Re-check from position i (but not before 0)
                i = max(0, i - 1)
        
        # Pass 2: Segment reversals (2-opt)
        for seg_len in [2, 3, 4]:
            for i in range(n_jobs - seg_len + 1):
                if (time.time() - start_time) >= time_limit:
                    break
                trial = best_order[:]
                trial[i:i+seg_len] = reversed(trial[i:i+seg_len])
                ms = _makespan(processing_times, trial)
                if ms < best_ms - 1e-10:
                    best_order = trial
                    best_ms = ms
                    improved = True
                    i = max(0, i - 1)
        
        # Pass 3: Insert moves (pick up a job and insert elsewhere)
        for i in range(n_jobs):
            if (time.time() - start_time) >= time_limit:
                break
            job = best_order[i]
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                trial = best_order[:]
                trial.pop(i)
                # Adjust j if i < j
                insert_pos = j if j > i else j
                trial.insert(insert_pos, job)
                ms = _makespan(processing_times, trial)
                if ms < best_ms - 1e-10:
                    best_order = trial
                    best_ms = ms
                    improved = True
                    break  # Restart pass with new best
    
    return best_order


def _simulated_annealing(processing_times, order, time_limit=10.0):
    """Simulated annealing to escape local optima."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    start_time = time.time()
    
    T = max(1.0, current_ms * 0.1)
    T_min = max(0.1, current_ms * 0.001)
    cooling = 0.995
    
    rng = random.Random(123)
    
    while (time.time() - start_time) < time_limit and T > T_min:
        # Generate neighbor: random swap or insert
        move_type = rng.random()
        if move_type < 0.5:
            # Swap two random positions
            i, j = rng.sample(range(n_jobs), 2)
            trial = current_order[:]
            trial[i], trial[j] = trial[j], trial[i]
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            trial = current_order[:]
            job = trial.pop(i)
            insert_pos = j if j > i else j
            trial.insert(insert_pos, job)
        
        trial_ms = _makespan(processing_times, trial)
        delta = trial_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_order = trial
            current_ms = trial_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        T *= cooling
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses multi-start heuristics, local search, and simulated annealing.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate initial solutions from multiple NEH variants
    initial_solutions = _neh_variants(processing_times)
    
    # Add LPT and SPT
    initial_solutions.append(list(np.argsort(-processing_times.sum(axis=1))))
    initial_solutions.append(list(np.argsort(processing_times.sum(axis=1))))
    
    # Add random permutations
    rng = random.Random(42)
    for _ in range(5):
        perm = list(range(n_jobs))
        rng.shuffle(perm)
        initial_solutions.append(perm)
    
    # Evaluate all initial solutions and find the best
    best_order = None
    best_ms = float('inf')
    for sol in initial_solutions:
        ms = _makespan(processing_times, sol)
        if ms < best_ms:
            best_ms = ms
            best_order = sol
    
    # Run local search on the best initial solution
    if (time.time() - start_time) < deadline - 10.0:
        time_remaining = deadline - 10.0 - (time.time() - start_time)
        time_remaining = max(time_remaining, 1.0)
        best_order = _local_search(processing_times, best_order, time_limit=time_remaining)
        best_ms = _makespan(processing_times, best_order)
    
    # Run local search on other initial solutions
    for sol in initial_solutions:
        if sol == best_order:
            continue
        if (time.time() - start_time) >= deadline - 10.0:
            break
        time_remaining = deadline - 10.0 - (time.time() - start_time)
        if time_remaining < 2.0:
            break
        candidate = _local_search(processing_times, sol, time_limit=time_remaining)
        ms = _makespan(processing_times, candidate)
        if ms < best_ms:
            best_ms = ms
            best_order = candidate
    
    # Use remaining time for simulated annealing to escape local optima
    if (time.time() - start_time) < deadline - 5.0:
        time_remaining = deadline - 5.0 - (time.time() - start_time)
        if time_remaining > 2.0:
            sa_order = _simulated_annealing(processing_times, best_order, time_limit=time_remaining)
            sa_ms = _makespan(processing_times, sa_order)
            if sa_ms < best_ms:
                best_ms = sa_ms
                best_order = sa_order
            
            # If time remains, run local search on the SA result
            if (time.time() - start_time) < deadline - 2.0:
                time_remaining = deadline - 2.0 - (time.time() - start_time)
                if time_remaining > 1.0:
                    final_order = _local_search(processing_times, best_order, time_limit=time_remaining)
                    final_ms = _makespan(processing_times, final_order)
                    if final_ms < best_ms:
                        best_ms = final_ms
                        best_order = final_order
    
    return list(best_order)
# EVOLVE-BLOCK-END
