"""Improved flow-shop scheduling via NEH heuristic with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    machine_completion = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in order:
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(prev_completion, machine_completion[m])
            prev_completion = start_time + processing_times[job_idx, m]
            machine_completion[m] = prev_completion
        if prev_completion > makespan:
            makespan = prev_completion
    
    return makespan


def _makespan_fast(processing_times, order):
    """Vectorized makespan computation for speed."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # Use a 2D array for completion times
    # completion[j][m] = completion time of job j on machine m
    completion = np.zeros((n_jobs, n_machines))
    
    for j_pos in range(n_jobs):
        job_idx = order[j_pos]
        for m in range(n_machines):
            if m == 0:
                # First machine: only depends on previous job on this machine
                if j_pos == 0:
                    completion[j_pos, m] = processing_times[job_idx, m]
                else:
                    completion[j_pos, m] = completion[j_pos - 1, m] + processing_times[job_idx, m]
            else:
                # Other machines: depends on previous machine for this job and previous job on this machine
                if j_pos == 0:
                    completion[j_pos, m] = completion[j_pos, m - 1] + processing_times[job_idx, m]
                else:
                    completion[j_pos, m] = max(completion[j_pos - 1, m], completion[j_pos, m - 1]) + processing_times[job_idx, m]
    
    return completion[-1, -1]


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling with efficient evaluation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position in the sequence
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Use a faster evaluation by only checking completion of last job
        for pos in range(len(sequence) + 1):
            # Build candidate and compute makespan efficiently
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _local_search(processing_times, initial_order, time_limit=10.0):
    """Or-opt local search with time limit."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try 2-opt swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Swap jobs at positions i and j
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break
        
        # Try or-opt moves (move a single job to a different position)
        if not improved:
            for i in range(n_jobs):
                if time.time() - start_time > time_limit:
                    break
                job_to_move = current_order[i]
                for j in range(n_jobs):
                    if j == i or j == i + 1:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    
                    # Remove job from position i and insert at position j
                    new_order = current_order[:i] + current_order[i+1:]
                    insert_pos = j if j > i else j
                    new_order = new_order[:insert_pos] + [job_to_move] + new_order[insert_pos:]
                    new_makespan = _makespan(processing_times, new_order)
                    
                    if new_makespan < current_makespan - 1e-9:
                        current_order = new_order
                        current_makespan = new_makespan
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_order = list(current_order)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_order


def _simulated_annealing(processing_times, initial_order, time_limit=30.0, rng=None):
    """Simulated annealing for flow shop scheduling."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    initial_temp = max(1.0, current_makespan * 0.1)
    cooling_rate = 0.9995
    
    temp = initial_temp
    no_improve_count = 0
    patience = 5000
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor by swapping two random positions
        i, j = rng.choice(n_jobs, 2, replace=False)
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_makespan = _makespan(processing_times, new_order)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_order = new_order
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        else:
            no_improve_count += 1
        
        temp *= cooling_rate
        
        if no_improve_count > patience:
            break
    
    return best_order


def _random_perturbation(order, strength=0.3, rng=None):
    """Apply random 2-opt perturbation."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    if rng is None:
        rng = np.random
    
    new_order = list(order)
    num_swaps = max(1, int(strength * n))
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    
    return new_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses NEH heuristic, local search, and simulated annealing with time management.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard: we must return well before 150s
    deadline = 140.0  # 10s safety margin
    start_time = time.time()
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan(processing_times, neh_order)
    
    best_order = list(neh_order)
    best_makespan = neh_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Local search on the NEH solution
    ls_time = min(remaining * 0.3, 30.0)
    ls_order = _local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: Simulated annealing with multiple restarts
    rng = np.random.RandomState(42)
    
    # Allocate remaining time to SA
    sa_time = remaining * 0.7
    if sa_time > 5.0:
        sa_order = _simulated_annealing(processing_times, best_order, time_limit=sa_time, rng=rng)
        sa_makespan = _makespan(processing_times, sa_order)
        
        if sa_makespan < best_makespan:
            best_order = list(sa_order)
            best_makespan = sa_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 3.0:
        return best_order
    
    # Step 4: Multiple restarts with perturbation and local search
    num_restarts = min(10, int(remaining / 3.0))
    
    for r in range(num_restarts):
        if time.time() + 3.0 > deadline:
            break
        
        # Perturb the best solution
        perturbed = _random_perturbation(best_order, strength=0.2 + 0.05 * r, rng=rng)
        
        # Local search on perturbed solution
        ls_time_r = min(2.0, deadline - time.time() - 1.0)
        if ls_time_r < 0.1:
            break
        
        ls_order_r = _local_search(processing_times, perturbed, time_limit=ls_time_r)
        ls_makespan_r = _makespan(processing_times, ls_order_r)
        
        if ls_makespan_r < best_makespan:
            best_order = list(ls_order_r)
            best_makespan = ls_makespan_r
    
    return best_order
# EVOLVE-BLOCK-END
