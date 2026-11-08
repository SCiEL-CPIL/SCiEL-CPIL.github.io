"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation of job indices."""
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            completion[m] = completion[m] + processing_times[job, m]
    return completion[-1]


def _incremental_makespan(perm, processing_times, i, j):
    """
    Compute makespan after swapping positions i and j in perm.
    Uses prefix and suffix decomposition for O(n) evaluation instead of O(n*m).
    Actually, let's use a full O(n*m) but optimized numpy version for simplicity and correctness first,
    then optimize if needed. Given n is likely small (<100), O(n*m) is fine.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    completion = np.zeros(n_machines)
    for k in range(n_jobs):
        job = perm[k]
        for m in range(n_machines):
            completion[m] += processing_times[job, m]
    return completion[-1]


def _swap_makespan(perm, i, j, processing_times):
    """
    Efficiently compute the makespan change when swapping two jobs at positions i and j.
    This is tricky for flow shop. Let's just recompute the whole thing but with numpy vectorization.
    """
    n_machines = processing_times.shape[1]
    # Create a new perm with swap
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return _makespan(new_perm, processing_times)


def _insert_makespan(perm, job, pos, processing_times):
    """
    Compute makespan if job is inserted at position pos in perm.
    """
    new_perm = perm[:pos] + [job] + perm[pos:]
    return _makespan(new_perm, processing_times)


def _simulated_annealing(initial_perm, processing_times, start_time, deadline):
    """
    Simulated Annealing for Flow Shop Scheduling.
    """
    n_jobs = len(initial_perm)
    n_machines = processing_times.shape[1]
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Temperature parameters
    t0 = 1000.0
    t_end = 1.0
    cooling_rate = 0.999  # Adjust based on number of iterations
    
    # Estimate number of iterations possible
    # Assume ~10000 iterations per second for small n
    # Let's just run until deadline
    
    t = t0
    no_improve_count = 0
    max_no_improve = 500  # Restart if stuck
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart with a random perturbation or break
            # For simplicity, let's just break and try another initial solution
            break
            
        # Choose a move: swap two random positions
        if n_jobs < 2:
            break
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
            
        new_perm = list(current_perm)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_makespan = _makespan(new_perm, processing_times)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < math.exp(-delta / max(t, 1e-10)):
            current_perm = new_perm
            current_makespan = new_makespan
            no_improve_count = 0
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
        else:
            no_improve_count += 1
            
        # Cool down
        t *= cooling_rate
        if t < t_end:
            t = t_end
            
        # Check time
        if time.time() > deadline - 0.5:
            break
            
    return best_perm, best_makespan


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using Simulated Annealing."""
    start_time = time.time()
    # Safety margin: stop searching at 140s, return before 145s
    search_deadline = start_time + 140.0
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate diverse initial orderings
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    spt_order = list(np.argsort(processing_times.sum(axis=1)))
    
    # Add a few random orders
    random.seed(42)
    random_orders = []
    for _ in range(5):
        o = list(range(n_jobs))
        random.shuffle(o)
        random_orders.append(o)
        
    initial_solutions = [lpt_order, spt_order] + random_orders
    
    best_perm = None
    best_makespan = float('inf')
    
    # Run SA from multiple starting points
    # Allocate time dynamically
    time_per_sa = 20.0  # 20 seconds per SA run
    
    for idx, init_order in enumerate(initial_solutions):
        if time.time() > search_deadline:
            break
            
        # Set deadline for this SA run
        sa_deadline = min(search_deadline, start_time + (idx + 1) * time_per_sa)
        
        sa_perm, sa_makespan = _simulated_annealing(
            init_order, processing_times, start_time, sa_deadline
        )
        
        if sa_makespan < best_makespan:
            best_makespan = sa_makespan
            best_perm = list(sa_perm)
            
    # If time remains, do additional SA runs or refine the best solution
    if best_perm is not None and time.time() < search_deadline:
        remaining_time = search_deadline - time.time()
        if remaining_time > 5.0:
            # Run more SA iterations from the best solution found
            sa_perm, sa_makespan = _simulated_annealing(
                best_perm, processing_times, start_time, search_deadline
            )
            if sa_makespan < best_makespan:
                best_makespan = sa_makespan
                best_perm = list(sa_perm)
    
    return best_perm if best_perm is not None else list(range(n_jobs))
# EVOLVE-BLOCK-END
