"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation using DP."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # completion[j][m] = max(completion[j-1][m], completion[j][m-1]) + pt[perm[j]][m]
    # We only need the previous row
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        for m in range(1, n_machines):
            cur_row[m] = max(prev_row[m], cur_row[m - 1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _neh_order(processing_times, rng):
    """
    NEH heuristic with some randomization.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Total processing time per job
    totals = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    # With some randomization: occasionally swap adjacent jobs in the sorted list
    order = np.argsort(-totals)
    
    # Add randomization: with probability proportional to difference, swap
    # Actually, let's do a proper NEH with insertion
    # Start with the first job
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        # Try all insertion positions
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _improve_order(perm, processing_times, time_limit, rng):
    """
    Simulated annealing with insert moves.
    """
    n_jobs = len(perm)
    if n_jobs <= 2:
        return perm, _makespan(perm, processing_times)
    
    n_machines = processing_times.shape[1]
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Initial temperature
    # Estimate based on makespan
    T = current_ms * 0.1
    T_min = current_ms * 0.001
    cooling = 0.995
    
    start_time = time.time()
    elapsed = 0.0
    
    while elapsed < time_limit:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Generate a neighbor: remove a job and insert it at another position
        i = rng.randint(n_jobs)
        j = rng.randint(n_jobs)
        if i == j:
            continue
        
        # Remove job at position i
        job = current_perm[i]
        new_perm = current_perm[:i] + current_perm[i+1:]
        # Insert at position j (adjust if j > i)
        if j > i:
            j -= 1
        new_perm = new_perm[:j+1] + [job] + new_perm[j+1:]
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Update elapsed time periodically
        if rng.random() < 0.01:
            elapsed = time.time() - start_time
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    """
    processing_times = np.asarray(processing_times)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline: 140 seconds (10s safety margin from 150s limit)
    deadline = 140.0
    start_time = time.time()
    
    # Create a reproducible RNG based on the input
    # Use a hash of the input to seed the RNG for reproducibility
    input_hash = hash(processing_tots.tobytes() if hasattr(processing_times, 'tobytes') else str(processing_times))
    rng = np.random.default_rng(input_hash % (2**32))
    
    # Phase 1: Generate multiple NEH variants and pick the best
    n_neh_variants = 5
    best_perm = None
    best_ms = float('inf')
    
    for _ in range(n_neh_variants):
        # Add some randomization to the NEH ordering
        # Shuffle with a bias towards the standard NEH order
        perm = _neh_order(processing_times, rng)
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
        
        # Check time
        if time.time() - start_time > deadline * 0.2:
            break
    
    if best_perm is None:
        # Fallback to simple ordering
        best_perm = list(np.argsort(-processing_times.sum(axis=1)))
        best_ms = _makespan(best_perm, processing_times)
    
    # Phase 2: Improve with simulated annealing
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 1.0:
        # Use most of the remaining time for SA, but leave a small buffer
        sa_time = remaining_time * 0.9
        best_perm, best_ms = _improve_order(best_perm, processing_times, sa_time, rng)
    
    # Phase 3: Local search refinement if time remains
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 0.5:
        # Do a few more rounds of 2-opt-like improvements
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            for i in range(n_jobs - 1):
                for j in range(i + 1, n_jobs):
                    # Try swapping i and j
                    new_perm = list(best_perm)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    new_ms = _makespan(new_perm, processing_times)
                    if new_ms < best_ms:
                        best_perm = new_perm
                        best_ms = new_ms
                        improved = True
                    # Check time
                    if (time.time() - start_time) > deadline:
                        break
                if (time.time() - start_time) > deadline:
                    break
    
    return list(best_perm)
# EVOLVE-BLOCK-END
