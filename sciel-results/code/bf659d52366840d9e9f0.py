"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan_from_order(pt, order):
    """Compute makespan for a given job order. O(n*m) using cumulative sums."""
    n_jobs = len(order)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    # Build the sequence of jobs
    seq = pt[order]  # shape (n_jobs, n_machines)
    
    # Use a single array for the current row of completions
    # comp[j] = completion time on machine j for the current job
    # After processing job i: comp_new[j] = max(comp_old[j], comp_new[j-1]) + p[i][j]
    
    comp = np.zeros(n_machines)
    for i in range(n_jobs):
        job_pt = seq[i]
        prev = 0.0
        for j in range(n_machines):
            prev = max(comp[j], prev) + job_pt[j]
            comp[j] = prev
    return comp[-1]


def _neh_initialization(pt, seed=None):
    """
    NEH heuristic with optional randomization for diversity.
    """
    n_jobs, n_machines = pt.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Sort jobs by total processing time descending, with optional perturbation
    totals = pt.sum(axis=1)
    
    # Add small random perturbation to break ties and create diversity
    range_totals = totals.max() - totals.min()
    if range_totals < 1e-9:
        range_totals = 1.0
        
    perturbation = rng.rand(n_jobs) * 0.05 * range_totals
    perturbed_totals = totals + perturbation
    
    sorted_jobs = np.argsort(-perturbed_totals)
    
    # Insert jobs one by one into the best position
    order = [sorted_jobs[0]]
    for idx in range(1, n_jobs):
        job = sorted_jobs[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            # Try inserting job at position pos
            new_order = order[:pos] + [job] + order[pos:]
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _vns_search(pt, initial_order, time_limit=145.0):
    """
    Variable Neighborhood Search with insertion neighborhoods of increasing radius.
    """
    n_jobs = len(initial_order)
    n_machines = pt.shape[1]
    
    if n_jobs <= 1:
        return initial_order
    
    best_order = list(initial_order)
    best_ms = _makespan_from_order(pt, best_order)
    
    start_time = time.time()
    max_k = min(n_jobs - 1, 50)  # Maximum neighborhood radius
    
    # VNS: iterate over neighborhood structures k = 1, 2, ..., max_k
    k = 1
    while k <= max_k:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        attempts = 0
        max_attempts_per_k = max(10, n_jobs * 2)
        
        while attempts < max_attempts_per_k:
            if time.time() - start_time > time_limit:
                return best_order
            attempts += 1
            
            # Random insertion: pick a random job and move it to a random position within radius k
            i = np.random.randint(0, n_jobs)
            max_shift = min(k, n_jobs - 1)
            j_offset = np.random.randint(-max_shift, max_shift + 1)
            j = i + j_offset
            if j < 0 or j >= n_jobs:
                continue
            if i == j:
                continue
            
            # Perform insertion: remove job at i, insert at j
            job = best_order[i]
            new_order = best_order[:i] + best_order[i+1:]
            if i < j:
                j -= 1
            new_order = new_order[:j] + [job] + new_order[j:]
            
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms - 1e-12:
                best_order = new_order
                best_ms = ms
                improved = True
                k = 1
                break
        
        if not improved:
            k += 1
    
    return best_order


def _simulated_annealing(pt, initial_order, time_limit=145.0):
    """
    Simulated annealing with insertion moves to escape local optima.
    """
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return initial_order
    
    current_order = list(initial_order)
    current_ms = _makespan_from_order(pt, current_order)
    
    best_order = list(current_order)
    best_ms = current_ms
    
    start_time = time.time()
    
    # Temperature parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    # Scale temperature by typical makespan magnitude
    scale = max(current_ms, 1.0)
    T_start *= scale
    T_end *= scale
    
    T = T_start
    rng = np.random.RandomState(42)
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Generate neighbor: random insertion
        i = rng.randint(0, n_jobs)
        j = rng.randint(0, n_jobs)
        if i == j:
            continue
        
        job = current_order[i]
        new_order = current_order[:i] + current_order[i+1:]
        if i < j:
            j -= 1
        new_order = new_order[:j] + [job] + new_order[j:]
        
        new_ms = _makespan_from_order(pt, new_order)
        delta = new_ms - current_ms
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (T > 1e-10 and rng.random() < np.exp(-delta / T)):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = list(current_order)
        
        # Cool down
        T *= cooling_rate
    
    return best_order


def _local_search_polish(pt, order, time_limit=145.0):
    """
    Final polish with first-improvement adjacent swaps and insertions.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan_from_order(pt, best_order)
    
    start_time = time.time()
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Try all adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                return best_order
            new_order = best_order.copy()
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
            ms = _makespan_from_order(pt, new_order)
            if ms < best_ms - 1e-12:
                best_order = new_order
                best_ms = ms
                improved = True
                break
        
        # Try insertions: move each job to each other position
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                return best_order
            job = best_order[i]
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                new_order = best_order[:i] + best_order[i+1:]
                if i < j:
                    j -= 1
                new_order = new_order[:j] + [job] + new_order[j:]
                ms = _makespan_from_order(pt, new_order)
                if ms < best_ms - 1e-12:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop instance.
    Uses multiple NEH restarts with VNS, simulated annealing, and local polish.
    """
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs = pt.shape[0]
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Set a deadline guard: 150s wall-clock limit, use 145s as safety margin
    start_time = time.time()
    time_limit = 145.0
    
    best_order = None
    best_ms = float('inf')
    
    # Determine number of restarts based on problem size
    # For small problems, more restarts; for large problems, fewer
    if n_jobs < 30:
        n_restarts = 15
    elif n_jobs < 50:
        n_restarts = 8
    else:
        n_restarts = 4
    
    for r in range(n_restarts):
        # Check if we have enough time left for a full iteration
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break
        
        # Allocate time budget for this restart
        # Reserve some time for the final polish
        if r == n_restarts - 1:
            # Last restart: give more time to local search
            restart_time = remaining * 0.7
            polish_time = remaining * 0.3
        else:
            # Earlier restarts: balanced allocation
            restart_time = remaining * 0.4
            polish_time = remaining * 0.3
        
        # Step 1: NEH initialization with randomization
        order = _neh_initialization(pt, seed=r * 12345 + 99)
        
        # Step 2: VNS search
        elapsed = time.time() - start_time
        vns_time = max(0.1, restart_time - (time.time() - start_time - elapsed))
        if vns_time > 0.5:
            order = _vns_search(pt, order, time_limit=vns_time)
        
        # Step 3: Simulated annealing for global escape
        elapsed = time.time() - start_time
        sa_time = max(0.1, restart_time - (time.time() - start_time - elapsed))
        if sa_time > 0.5:
            order = _simulated_annealing(pt, order, time_limit=sa_time)
        
        # Step 4: Local search polish
        elapsed = time.time() - start_time
        polish_time = max(0.1, time_limit - (time.time() - start_time))
        order = _local_search_polish(pt, order, time_limit=polish_time)
        
        # Update best
        ms = _makespan_from_order(pt, order)
        if ms < best_ms:
            best_ms = ms
            best_order = order
    
    if best_order is None:
        best_order = _neh_initialization(pt)
    
    return best_order
# EVOLVE-BLOCK-END
