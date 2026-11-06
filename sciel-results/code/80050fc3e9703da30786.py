"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation efficiently."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times[-1]


def _compute_delta_table(processing_times):
    """
    Precompute delta costs for adjacent swaps.
    For a permutation, swapping adjacent jobs at positions i and i+1 changes the makespan.
    We precompute pairwise interaction costs to enable O(1) delta evaluation.
    
    The key insight: for flow shop, the completion time on the last machine depends on
    the order of jobs. We can precompute for each pair of adjacent jobs (j1, j2) the
    change in makespan if they are swapped.
    
    Returns a function that computes delta for swapping positions i and i+1 in a given perm.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Precompute: for each pair of jobs (a, b), compute the "interaction cost"
    # when a precedes b vs b precedes a
    # This is complex for flow shop, so we use a different approach:
    # We'll compute the delta by tracking the completion times incrementally.
    
    # For now, we'll use a vectorized approach to compute makespan deltas
    # by maintaining completion time arrays and updating only affected positions.
    
    def _makespan_with_prefix(perm, processing_times, start_pos=0):
        """Compute completion times starting from position start_pos."""
        n_machines = processing_times.shape[1]
        machine_times = np.zeros(n_machines, dtype=np.float64)
        
        # Process jobs from start_pos to end
        for j in range(start_pos, len(perm)):
            job = perm[j]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(machine_times[m], prev) + pt[m]
                machine_times[m] = new_time
                prev = new_time
        
        return machine_times
    
    return _makespan_with_prefix


def _delta_swap_adjacent(perm, i, processing_times, completion_before):
    """
    Compute the delta in makespan when swapping adjacent jobs at positions i and i+1.
    
    Args:
        perm: current permutation
        i: position of first job in swap
        processing_times: processing times matrix
        completion_before: completion times array before position i
    
    Returns:
        delta: change in makespan (new_makespan - old_makespan)
    """
    n_machines = processing_times.shape[1]
    
    # Get the two jobs to swap
    job1 = perm[i]
    job2 = perm[i + 1]
    
    # Compute completion times for the rest of the schedule starting from position i+2
    # First, compute what happens if we keep the order (job1 then job2)
    mt_old = completion_before.copy()
    prev = 0.0
    for m in range(n_machines):
        new_time = max(mt_old[m], prev) + processing_times[job1][m]
        mt_old[m] = new_time
        prev = new_time
    prev = 0.0
    for m in range(n_machines):
        new_time = max(mt_old[m], prev) + processing_times[job2][m]
        mt_old[m] = new_time
        prev = new_time
    
    # Compute what happens if we swap (job2 then job1)
    mt_new = completion_before.copy()
    prev = 0.0
    for m in range(n_machines):
        new_time = max(mt_new[m], prev) + processing_times[job2][m]
        mt_new[m] = new_time
        prev = new_time
    prev = 0.0
    for m in range(n_machines):
        new_time = max(mt_new[m], prev) + processing_times[job1][m]
        mt_new[m] = new_time
        prev = new_time
    
    # Now we need to compute the makespan for the rest of the jobs starting from position i+2
    # This is where it gets expensive. For now, we'll approximate by only considering
    # the change in the completion times at position i+1.
    
    # For a more accurate delta, we'd need to propagate through all remaining jobs.
    # This is O(n*m) in the worst case, but for small n it's fine.
    
    # Let's compute the full makespan for both cases
    def _compute_remaining_makespan(machine_times, perm, start_pos):
        n_machines = processing_times.shape[1]
        mt = machine_times.copy()
        for j in range(start_pos, len(perm)):
            job = perm[j]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(mt[m], prev) + pt[m]
                mt[m] = new_time
                prev = new_time
        return mt[-1]
    
    old_ms = _compute_remaining_makespan(mt_old, perm, i + 2)
    new_ms = _compute_remaining_makespan(mt_new, perm, i + 2)
    
    return new_ms - old_ms


def _adjacent_swap_local_search(perm, processing_times, time_limit=5.0):
    """
    Local search using only adjacent swaps with delta evaluation.
    This is more efficient than 2-opt swaps because adjacent swaps have
    simpler delta computations.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            
            # Compute completion times up to position i
            mt_before = np.zeros(processing_times.shape[1], dtype=np.float64)
            for j in range(i):
                job = best_perm[j]
                pt = processing_times[job]
                prev = 0.0
                for m in range(processing_times.shape[1]):
                    new_time = max(mt_before[m], prev) + pt[m]
                    mt_before[m] = new_time
                    prev = new_time
            
            # Compute delta for swapping positions i and i+1
            delta = _delta_swap_adjacent(best_perm, i, processing_times, mt_before)
            
            if delta < -1e-9:
                # Accept the swap
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms += delta
                improved = True
    
    return best_perm, best_ms


def _neh_order(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert jobs one by one into the best position
    perm = []
    for job in sorted_jobs:
        # Try inserting at each position
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


def _neighborhood_search(perm, processing_times, time_limit=5.0):
    """Improve permutation using 2-opt style neighborhood search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, n_jobs):
                # Swap jobs at positions i and j
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, processing_times)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
    
    return best_perm, best_ms


def _simulated_annealing(perm, processing_times, time_limit=10.0):
    """Simulated annealing for further improvement."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    start_time = time.time()
    t0 = 100.0
    t_end = 1.0
    cooling_rate = 0.995
    
    temperature = t0
    while temperature > t_end and (time.time() - start_time) < time_limit:
        # Generate a random move: swap two random positions
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
        
        candidate = list(current_perm)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_ms = _makespan(candidate, processing_times)
        
        delta = candidate_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_perm = candidate
            current_ms = candidate_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best_perm = list(current_perm)
        
        temperature *= cooling_rate
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic followed by adjacent-swap local search, 2-opt search, and simulated annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (120s limit, use 110s)
    global_deadline = time.time() + 110.0
    
    # Step 1: NEH heuristic
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    # Step 2: Adjacent swap local search (faster due to delta evaluation)
    adj_time_limit = min(5.0, max(0.5, (global_deadline - time.time()) * 0.2))
    adj_perm, adj_ms = _adjacent_swap_local_search(neh_perm, processing_times, time_limit=adj_time_limit)
    
    # Step 3: Neighborhood search (2-opt swaps)
    remaining_after_adj = global_deadline - time.time()
    ns_time_limit = min(10.0, max(1.0, remaining_after_adj * 0.3))
    ns_perm, ns_ms = _neighborhood_search(adj_perm, processing_times, time_limit=ns_time_limit)
    
    # Step 4: Simulated annealing if time remains
    remaining_time = global_deadline - time.time()
    if remaining_time > 2.0:
        sa_time_limit = min(remaining_time - 1.0, 30.0)
        sa_perm, sa_ms = _simulated_annealing(ns_perm, processing_times, time_limit=sa_time_limit)
        if sa_ms < ns_ms:
            return sa_perm
    
    return ns_perm
# EVOLVE-BLOCK-END
