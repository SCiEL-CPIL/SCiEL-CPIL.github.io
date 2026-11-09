"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # C[i][j] = completion time of job i on machine j
    # We can compute this in O(n_jobs * n_machines)
    completion = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                if i == 0:
                    completion[i, j] = p
                else:
                    completion[i, j] = completion[i-1, j] + p
            else:
                if i == 0:
                    completion[i, j] = completion[i, j-1] + p
                else:
                    completion[i, j] = max(completion[i-1, j], completion[i, j-1]) + p
    
    return completion[-1, -1]

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using Simulated Annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Time management
    start_time = time.time()
    max_time = 110.0  # Leave 10s buffer
    
    # Initialize with random permutation
    rng = np.random.RandomState(42)
    current_order = list(rng.permutation(n_jobs))
    current_ms = _makespan(processing_times, current_order)
    
    best_order = list(current_order)
    best_ms = current_ms
    
    # SA Parameters
    # Heuristic for initial temperature: average delta / 0.1
    # We estimate average delta by sampling a few random moves
    avg_delta = 0.0
    samples = 10
    for _ in range(samples):
        o1, o2 = rng.choice(n_jobs, 2, replace=False)
        neighbor = list(current_order)
        # Simple swap for delta estimation
        neighbor[o1], neighbor[o2] = neighbor[o2], neighbor[o1]
        delta = _makespan(processing_times, neighbor) - current_ms
        avg_delta += abs(delta)
    avg_delta /= samples
    if avg_delta < 1e-6:
        avg_delta = 1.0
        
    T = avg_delta / 0.1  # Start high enough
    alpha = 0.995  # Cooling rate
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < max_time:
        # Generate neighbor: Insertion move
        i = rng.randint(n_jobs)
        j = rng.randint(n_jobs)
        
        if i == j:
            continue
            
        neighbor = list(current_order)
        job_to_move = neighbor.pop(i)
        # Adjust index if i < j
        if i < j:
            j -= 1
        neighbor.insert(j, job_to_move)
        
        new_ms = _makespan(processing_times, neighbor)
        delta = new_ms - current_ms
        
        if delta < 0:
            # Better solution
            current_order = neighbor
            current_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_order = list(neighbor)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Worse solution: accept with probability exp(-delta/T)
            if T > 1e-6:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    current_order = neighbor
                    current_ms = new_ms
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        # Cool down
        T *= alpha
        
        # Early stopping if no improvement for long and T is low
        if no_improve_count > max_no_improve and T < 1e-4:
            break
            
        iteration += 1
        
    return best_order
# EVOLVE-BLOCK-END
