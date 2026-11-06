"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # C[j, m] = completion time of job j on machine m
    # We process jobs in the order given by 'order'
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + P[j][m]
    # where C[j-1] refers to the previous job in the sequence
    
    # For efficiency, we track only the previous job's completion times
    prev_completions = np.zeros(n_machines)
    current_completions = np.zeros(n_machines)
    
    for job_idx in order:
        for m in range(n_machines):
            if m == 0:
                current_completions[m] = prev_completions[m] + processing_times[job_idx, m]
            else:
                current_completions[m] = max(current_completions[m-1], prev_completions[m]) + processing_times[job_idx, m]
        prev_completions = current_completions.copy()
    
    return prev_completions[-1]


def _makespan_delta_insert(processing_times, order, old_pos, new_pos):
    """
    Compute the makespan change when moving a job from old_pos to new_pos.
    This is an O(n_machines * |old_pos - new_pos|) operation.
    Returns the new makespan, or None if invalid.
    """
    n_jobs, n_machines = processing_times.shape
    if old_pos == new_pos:
        return _makespan(processing_times, order)
    
    # Create the new order
    job = order[old_pos]
    new_order = list(order)
    del new_order[old_pos]
    if old_pos < new_pos:
        new_order.insert(new_pos - 1, job)
    else:
        new_order.insert(new_pos, job)
    
    return _makespan(processing_times, new_order)


def _neh_initialization(processing_times, rng):
    """
    NEH (Newell and Muntz) heuristic for initial solution.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Build solution incrementally
    current_order = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        # Try all positions and pick the one with minimum makespan
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_order) + 1):
            trial = current_order[:pos] + [job] + current_order[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_order.insert(best_pos, job)
    
    return current_order


def _local_search(processing_times, order, max_iters=1000, deadline=None):
    """
    First-improvement insertion-based local search.
    Tries to move each job to a better position.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = current_order[i]
            for j in range(n_jobs):
                if i == j:
                    continue
                # Try moving job from i to j
                new_order = list(current_order)
                del new_order[i]
                if i < j:
                    new_order.insert(j - 1, job)
                else:
                    new_order.insert(j, job)
                
                new_ms = _makespan(processing_times, new_order)
                if new_ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    break  # Restart the scan after improvement
            if improved:
                break
    
    return current_order


def _vns_search(processing_times, order, max_k=5, max_rounds=10, deadline=None):
    """
    Variable Neighborhood Search.
    Systematically increases the neighborhood radius k.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    for k in range(1, max_k + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        found_improvement = False
        for _ in range(max_rounds):
            if deadline is not None and time.time() > deadline:
                break
            
            # Generate a random insertion within radius k
            i = np.random.randint(0, n_jobs)
            # Choose j such that |i - j| <= k
            min_j = max(0, i - k)
            max_j = min(n_jobs - 1, i + k)
            j = np.random.randint(min_j, max_j + 1)
            
            if i == j:
                continue
            
            job = current_order[i]
            new_order = list(current_order)
            del new_order[i]
            if i < j:
                new_order.insert(j - 1, job)
            else:
                new_order.insert(j, job)
            
            new_ms = _makespan(processing_times, new_order)
            if new_ms < current_ms - 1e-9:
                current_order = new_order
                current_ms = new_ms
                found_improvement = True
        
        if not found_improvement:
            break
    
    return current_order


def order_jobs(processing_times):
    """
    Main entry point. Uses multiple NEH restarts with local search and VNS.
    """
    processing_times = np.asarray(processing_times)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_order = None
    best_ms = float('inf')
    
    # Set random seed for reproducibility within a call
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    max_restarts = 50
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:  # Leave 2s buffer for final evaluation
            break
        
        # Generate initial solution using NEH with slight randomization
        if restart == 0:
            order = _neh_initialization(processing_times, rng)
        else:
            # Perturb the best solution found so far, or generate a new NEH
            if best_order is not None and rng.random() < 0.7:
                # Perturb best solution
                order = list(best_order)
                n_perturb = max(2, n_jobs // 10)
                for _ in range(n_perturb):
                    i = rng.integers(0, n_jobs)
                    j = rng.integers(0, n_jobs)
                    if i != j:
                        job = order[i]
                        del order[i]
                        if i < j:
                            order.insert(j - 1, job)
                        else:
                            order.insert(j, job)
            else:
                # Fresh NEH
                order = _neh_initialization(processing_times, rng)
        
        # Local search
        order = _local_search(processing_times, order, max_iters=500, deadline=deadline)
        
        # VNS
        order = _vns_search(processing_times, order, max_k=4, max_rounds=5, deadline=deadline)
        
        # Evaluate
        ms = _makespan(processing_times, order)
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
        
        restarts_done += 1
    
    if best_order is None:
        # Fallback
        best_order = _neh_initialization(processing_times, rng)
    
    return best_order
# EVOLVE-BLOCK-END
