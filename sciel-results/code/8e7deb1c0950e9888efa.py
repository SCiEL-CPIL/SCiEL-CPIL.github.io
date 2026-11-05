# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence.
    pt: (n, m) array of processing times.
    perm: array of job indices in order.
    Returns makespan (scalar).
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    if m == 0:
        return 0.0
    
    # comp[mi] stores the completion time of the current job on machine mi
    comp = np.zeros(m, dtype=float)
    
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            # Completion time on machine mi for job j
            # C[j][mi] = max(C[j-1][mi], C[j][mi-1]) + P[j][mi]
            # comp[mi] currently holds C[j-1][mi]
            # comp[mi-1] currently holds C[j][mi-1] (updated in previous iteration of mi)
            if mi > 0:
                prev_machine = comp[mi - 1]
            else:
                prev_machine = 0.0
            
            same_machine_prev = comp[mi]
            comp[mi] = max(same_machine_prev, prev_machine) + pt_row[mi]
    
    return comp[-1]

def _neh_initial(pt):
    """NEH heuristic: sort jobs by decreasing total processing time, then insert
    each job at the position that minimizes makespan of the partial schedule."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    
    # Sort by decreasing total processing time
    total_pt = pt.sum(axis=1)
    order = np.argsort(-total_pt).astype(int)
    
    # Build schedule by inserting each job at the best position
    schedule = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        # Optimization: Only check positions around the middle or use full scan
        # For small n, full scan is fine.
        for pos in range(len(schedule) + 1):
            new_schedule = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(np.array(new_schedule), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    
    return np.array(schedule, dtype=int)

def _palmer_initial(pt):
    """Palmer's heuristic: sort by decreasing key."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _bottleneck_initial(pt):
    """Sort by decreasing processing time on the bottleneck machine."""
    n, m = pt.shape
    if m == 0:
        return np.arange(n)
    machine_totals = pt.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    return np.argsort(-pt[:, bottleneck_machine]).astype(int)

def _random_initial(n, rng):
    """Random permutation."""
    return rng.permutation(n)

def _local_search_insertion(perm, pt, time_limit, start_time):
    """Best-improvement insertion local search.
    Tries to move each job to a better position.
    """
    n = len(perm)
    if n <= 1:
        return perm.copy(), _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_move_i = -1
        best_move_j = -1
        best_move_ms = best_ms
        
        # Scan all possible insertion moves
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            # Remove job i
            remaining = np.delete(best_perm, i)
            # Try inserting at each position j in the remaining sequence
            # j ranges from 0 to len(remaining)
            # Note: inserting at i is equivalent to no move (or shift), usually we skip i
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                new_perm = np.insert(remaining, j, job)
                ms = _makespan(new_perm, pt)
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
        
        if best_move_i >= 0:
            # Apply the best move
            job = best_perm[best_move_i]
            remaining = np.delete(best_perm, best_move_i)
            best_perm = np.insert(remaining, best_move_j, job)
            best_ms = best_move_ms
            improved = True
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initial solutions
    candidates = []
    
    # NEH
    candidates.append(_neh_initial(processing_times))
    
    # Palmer
    candidates.append(_palmer_initial(processing_times))
    
    # Bottleneck
    candidates.append(_bottleneck_initial(processing_times))
    
    # Random starts (a few)
    num_random = 5
    for _ in range(num_random):
        candidates.append(_random_initial(n, rng))
    
    # 2. Apply Local Search to each candidate
    # We allocate time based on the number of candidates
    # Let's give each candidate a portion of the time, but prioritize the best ones
    # Strategy: Run LS on all, keep the best. Then use remaining time for ILS on the best.
    
    best_perm = None
    best_ms = float('inf')
    
    # Time budget per candidate for initial LS
    # If n is small, LS is fast. If n is large, LS is slow.
    # We'll use a fixed number of iterations or a time slice.
    # Let's use a time slice of (deadline - 20) / len(candidates) to leave 20s for ILS
    
    time_per_candidate = max(1.0, (deadline - 20.0) / len(candidates))
    
    for perm in candidates:
        if time.time() - start_time > deadline - 20.0:
            break
            
        current_start = time.time()
        current_limit = min(time_per_candidate, deadline - 20.0 - (current_start - start_time))
        if current_limit <= 0:
            break
            
        new_perm, new_ms = _local_search_insertion(perm, processing_times, current_limit, current_start)
        
        if new_ms < best_ms:
            best_ms = new_ms
            best_perm = new_perm.copy()
    
    # 3. Iterated Local Search (ILS) on the best solution
    # Use remaining time to perturb the best solution and re-optimize
    remaining_time = deadline - (time.time() - start_time)
    
    if best_perm is not None and remaining_time > 5.0:
        current_perm = best_perm.copy()
        current_ms = best_ms
        
        # Perturbation strength: number of swaps or insertions
        # Start with small perturbations, increase if stuck?
        # For simplicity, use a fixed moderate perturbation: 2-3 random insertions
        
        while time.time() - start_time < deadline - 2.0:
            # Perturb
            perturbed_perm = current_perm.copy()
            
            # Perform 2 random insertion moves
            for _ in range(2):
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                job = perturbed_perm[i]
                remaining = np.delete(perturbed_perm, i)
                # Adjust j if i < j
                insert_j = j - 1 if j > i else j
                if insert_j < 0:
                    insert_j = 0
                if insert_j > len(remaining):
                    insert_j = len(remaining)
                perturbed_perm = np.insert(remaining, insert_j, job)
            
            # Run Local Search on perturbed solution
            ls_start = time.time()
            # Allocate a portion of remaining time to this LS
            # Heuristic: give it 10% of remaining time, or a minimum of 1s
            ls_time_budget = min(remaining_time * 0.1, 10.0)
            ls_time_budget = max(1.0, ls_time_budget)
            
            new_perm, new_ms = _local_search_insertion(perturbed_perm, processing_times, ls_time_budget, ls_start)
            
            if new_ms < current_ms:
                current_ms = new_ms
                current_perm = new_perm.copy()
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_perm = new_perm.copy()
            
            # Update remaining time
            remaining_time = deadline - (time.time() - start_time)
            if remaining_time < 5.0:
                break
            
            # If no improvement for several iterations, increase perturbation?
            # For now, keep it simple.
    
    return list(best_perm)

# EVOLVE-BLOCK-END
