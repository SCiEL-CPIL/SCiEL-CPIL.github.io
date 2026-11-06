"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using vectorized DP."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # completion[m] tracks the completion time on machine m after processing jobs so far
    # For each job in perm, we update completion times
    # completion[m] = max(completion[m-1], completion[m]) + pt[perm[j]][m]
    # This is sequential over machines, but we can vectorize over jobs if we think differently
    
    # Actually, let's use the standard DP:
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    # We can compute row by row
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    
    return prev_row[-1]


def _makespan_fast(perm, processing_times):
    """Faster makespan computation using numpy operations where possible."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # For small problems, the loop approach is fine
    # Let's try a different approach: compute the cumulative sums and use max
    # Actually, the sequential nature makes it hard to fully vectorize
    # Let's stick with the loop but optimize it
    
    prev = 0.0
    machine_times = np.zeros(n_machines)
    
    for job in perm:
        pt = processing_times[job]
        for m in range(n_machines):
            machine_times[m] = max(machine_times[m], prev) + pt[m]
            prev = machine_times[m]
    
    return prev


def _neh_order(processing_times):
    """Construct initial order using NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    
    # Compute total processing time for each job
    totals = processing_times.sum(axis=1)
    
    # Sort jobs by decreasing total processing time
    order = list(np.argsort(-totals))
    
    # NEH insertion procedure
    schedule = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(schedule) + 1):
            candidate = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan_fast(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        schedule = schedule[:best_pos] + [job] + schedule[best_pos:]
    
    return schedule


def _random_order(n_jobs):
    """Generate a random permutation."""
    return list(np.random.permutation(n_jobs))


def _swap_neighbor(perm, i, j):
    """Swap two adjacent elements."""
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm


def _improve_order(perm, processing_times, time_limit, max_iterations=10000):
    """Improve order using 2-opt style neighborhood search."""
    n_jobs = len(perm)
    current_ms = _makespan_fast(perm, processing_times)
    best_perm = list(perm)
    best_ms = current_ms
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while iteration < max_iterations:
        # Check time
        if time.time() - start_time > time_limit * 0.8:
            break
        
        # Pick a random pair of positions to swap
        if n_jobs < 2:
            break
        
        i = np.random.randint(0, n_jobs - 1)
        j = i + 1  # Adjacent swap for efficiency
        
        new_perm = _swap_neighbor(perm, i, j)
        new_ms = _makespan_fast(new_perm, processing_times)
        
        if new_ms < current_ms:
            perm = new_perm
            current_ms = new_ms
            no_improve_count = 0
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(perm)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        
        iteration += 1
    
    return best_perm, best_ms


def _multi_start(processing_times, time_limit):
    """Multi-start local search."""
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    
    best_perm = None
    best_ms = float('inf')
    
    # Try NEH first
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan_fast(neh_perm, processing_times)
    if neh_ms < best_ms:
        best_ms = neh_ms
        best_perm = list(neh_perm)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.7:
        return best_perm
    
    # Multi-start with random permutations
    n_starts = max(1, int((time_limit - elapsed) / 0.05))  # Estimate number of starts
    n_starts = min(n_starts, 50)  # Cap to avoid too many
    
    for start_idx in range(n_starts):
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # Generate random initial solution
        random_perm = _random_order(n_jobs)
        
        # Improve it
        improved_perm, improved_ms = _improve_order(
            random_perm, processing_times,
            time_limit * 0.05  # Small time budget per start
        )
        
        if improved_ms < best_ms:
            best_ms = improved_ms
            best_perm = improved_perm
    
    return best_perm


def order_jobs(processing_times):
    """Main entry point: construct job permutation minimizing makespan."""
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Set time limit with safety margin (150s total, use 140s for safety)
    total_time_limit = 140.0
    start_time = time.time()
    
    # Multi-start local search
    result = _multi_start(processing_times, total_time_limit)
    
    # If result is None (shouldn't happen), fall back to NEH
    if result is None:
        result = _neh_order(processing_times)
    
    # Final safety check
    elapsed = time.time() - start_time
    if elapsed > 145:  # Very close to limit
        # Return current best immediately
        return list(result)
    
    return list(result)
# EVOLVE-BLOCK-END
