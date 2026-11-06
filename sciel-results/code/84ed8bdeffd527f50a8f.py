"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given permutation of jobs."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _neh_initialization(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # Build solution incrementally
    order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        # Try all positions in the current order
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            trial = order[:pos] + [job] + order[pos:]
            ms = _makespan(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _improve_with_local_search(order, processing_times, max_iterations=500, time_limit=10.0):
    """Multi-neighborhood local search with 2-opt and insertion moves."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 1:
        return order
    
    current_order = list(order)
    current_ms = _makespan(current_order, processing_times)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        best_neighbor = None
        best_neighbor_ms = current_ms
        found_improvement = False
        
        # 2-opt moves: reverse a segment
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                if j - i < 2:  # Skip trivial reversals
                    continue
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                ms = _makespan(new_order, processing_times)
                if ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor = new_order
                    found_improvement = True
        
        # Insertion moves: move a job to a different position
        for i in range(n_jobs):
            job = current_order[i]
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                new_order = current_order[:i] + current_order[i+1:]
                if j > i:
                    j -= 1
                new_order = new_order[:j] + [job] + new_order[j:]
                ms = _makespan(new_order, processing_times)
                if ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor = new_order
                    found_improvement = True
        
        if found_improvement:
            current_order = best_neighbor
            current_ms = best_neighbor_ms
        else:
            break
    
    return current_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH initialization
    and multi-neighborhood local search with random restarts.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    overall_start = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    best_order = None
    best_ms = float('inf')
    
    # Phase 1: NEH initialization
    neh_order = _neh_initialization(processing_times)
    neh_ms = _makespan(neh_order, processing_times)
    
    if neh_ms < best_ms:
        best_ms = neh_ms
        best_order = list(neh_order)
    
    # Phase 2: Local search on NEH solution
    improved_order = _improve_with_local_search(
        neh_order, processing_times, 
        max_iterations=300, 
        time_limit=20.0
    )
    improved_ms = _makespan(improved_order, processing_times)
    
    if improved_ms < best_ms:
        best_ms = improved_ms
        best_order = list(improved_order)
    
    # Check time
    if time.time() - overall_start > deadline:
        return best_order
    
    # Phase 3: Random restarts with perturbation
    rng = np.random.default_rng(42)
    restart_count = 0
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - overall_start > deadline - 10:  # Stop if only 10s left
            break
        
        # Create a perturbed version of the best solution
        if restart == 0:
            # Start with NEH again
            current = list(neh_order)
        else:
            # Perturb the best solution
            current = list(best_order)
            n_perturb = max(2, n_jobs // 4)
            for _ in range(n_perturb):
                i = rng.integers(0, n_jobs)
                j = rng.integers(0, n_jobs)
                current[i], current[j] = current[j], current[i]
        
        # Local search on perturbed solution
        time_remaining = deadline - (time.time() - overall_start)
        if time_remaining < 5:
            break
        
        search_time = max(1.0, min(10.0, time_remaining * 0.3))
        improved = _improve_with_local_search(
            current, processing_times,
            max_iterations=200,
            time_limit=search_time
        )
        ms = _makespan(improved, processing_times)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(improved)
        
        restart_count += 1
    
    return best_order
# EVOLVE-BLOCK-END
