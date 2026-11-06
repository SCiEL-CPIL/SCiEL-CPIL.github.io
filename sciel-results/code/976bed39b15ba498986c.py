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
    
    # Vectorized computation of completion times
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # Using the recurrence: C[j,m] = C[j-1,m] + P[j,m] + max(0, C[j,m-1] - C[j-1,m])
    
    # Start with first machine: C[j, 0] = sum_{k<=j} P[k, 0]
    first_col = np.cumsum(processing_times[perm, 0])
    
    if n_machines == 1:
        return first_col[-1]
    
    # Process remaining machines
    prev_col = first_col.copy()
    current_col = first_col.copy()
    
    for m in range(1, n_machines):
        # C[j, m] = C[j-1, m] + P[j, m] + max(0, C[j, m-1] - C[j-1, m])
        # For j=0: C[0, m] = C[0, m-1] + P[0, m]
        # For j>0: C[j, m] = C[j-1, m] + P[j, m] + max(0, C[j, m-1] - C[j-1, m])
        
        p_col = processing_times[perm, m]
        
        # Compute the delay term: max(0, C[j, m-1] - C[j-1, m])
        # For j=0, C[j-1, m] doesn't exist, but C[0, m] = C[0, m-1] + P[0, m]
        # So we handle j=0 separately or use a trick
        
        # Let's compute iteratively but vectorize the inner part
        # Actually, let's just use the loop but with numpy for the max
        
        current_col[0] = prev_col[0] + p_col[0]
        for j in range(1, n_jobs):
            delay = max(0.0, current_col[j-1] - prev_col[j-1])
            current_col[j] = prev_col[j-1] + p_col[j] + delay
        
        prev_col = current_col.copy()
    
    return prev_col[-1]


def _compute_prefix_completions(perm, processing_times):
    """
    Precompute prefix completion times for all positions.
    prefix[i][m] = completion time on machine m after processing first i jobs.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix = np.zeros((n_jobs + 1, n_machines), dtype=np.float64)
    
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(prefix[i][m], prev) + pt[m]
            prefix[i + 1][m] = new_time
            prev = new_time
    
    return prefix


def _compute_suffix_makespan(perm, processing_times, start_pos, machine_times):
    """
    Compute makespan from start_pos onwards given initial machine times.
    """
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


def _fast_adjacent_swap_local_search(perm, processing_times, time_limit=5.0):
    """
    Fast local search using adjacent swaps with precomputed prefix completions.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Precompute prefix completions for current permutation
        prefix = _compute_prefix_completions(best_perm, processing_times)
        
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            
            job1 = best_perm[i]
            job2 = best_perm[i + 1]
            
            # State before position i
            mt_before = prefix[i]
            
            # Compute state after job1 then job2 (original order)
            mt_orig = mt_before.copy()
            prev = 0.0
            for m in range(n_machines):
                new_time = max(mt_orig[m], prev) + processing_times[job1][m]
                mt_orig[m] = new_time
                prev = new_time
            prev = 0.0
            for m in range(n_machines):
                new_time = max(mt_orig[m], prev) + processing_times[job2][m]
                mt_orig[m] = new_time
                prev = new_time
            
            # Compute state after job2 then job1 (swapped order)
            mt_swap = mt_before.copy()
            prev = 0.0
            for m in range(n_machines):
                new_time = max(mt_swap[m], prev) + processing_times[job2][m]
                mt_swap[m] = new_time
                prev = new_time
            prev = 0.0
            for m in range(n_machines):
                new_time = max(mt_swap[m], prev) + processing_times[job1][m]
                mt_swap[m] = new_time
                prev = new_time
            
            # If states are identical, no change in makespan
            if np.array_equal(mt_orig, mt_swap):
                continue
            
            # Compute remaining makespan for both cases
            old_ms = _compute_suffix_makespan(best_perm, processing_times, i + 2, mt_orig)
            new_ms = _compute_suffix_makespan(best_perm, processing_times, i + 2, mt_swap)
            
            delta = new_ms - old_ms
            if delta < -1e-9:
                best_perm[i], best_perm[i + 1] = best_perm[i + 1], best_perm[i]
                best_ms += delta
                improved = True
                # Recompute prefix since permutation changed
                break
    
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


def _polish_pass(perm, processing_times, time_limit=10.0):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    # Use very tight tolerance
    tol = 1e-12
    
    # Multiple passes of adjacent swap local search
    for _ in range(5):
        if (time.time() - start_time) >= time_limit:
            break
        
        improved = False
        prefix = _compute_prefix_completions(best_perm, processing_times)
        
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            
            job1 = best_perm[i]
            job2 = best_perm[i + 1]
            mt_before = prefix[i]
            
            # Compute state after job1 then job2
            mt_orig = mt_before.copy()
            prev = 0.0
            for m in range(processing_times.shape[1]):
                new_time = max(mt_orig[m], prev) + processing_times[job1][m]
                mt_orig[m] = new_time
                prev = new_time
            prev = 0.0
            for m in range(processing_times.shape[1]):
                new_time = max(mt_orig[m], prev) + processing_times[job2][m]
                mt_orig[m] = new_time
                prev = new_time
            
            # Compute state after job2 then job1
            mt_swap = mt_before.copy()
            prev = 0.0
            for m in range(processing_times.shape[1]):
                new_time = max(mt_swap[m], prev) + processing_times[job2][m]
                mt_swap[m] = new_time
                prev = new_time
            prev = 0.0
            for m in range(processing_times.shape[1]):
                new_time = max(mt_swap[m], prev) + processing_times[job1][m]
                mt_swap[m] = new_time
                prev = new_time
            
            if np.array_equal(mt_orig, mt_swap):
                continue
            
            old_ms = _compute_suffix_makespan(best_perm, processing_times, i + 2, mt_orig)
            new_ms = _compute_suffix_makespan(best_perm, processing_times, i + 2, mt_swap)
            
            delta = new_ms - old_ms
            if delta < -tol:
                best_perm[i], best_perm[i + 1] = best_perm[i + 1], best_perm[i]
                best_ms += delta
                improved = True
                break
        
        if not improved:
            break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic followed by fast adjacent-swap local search, 2-opt search, simulated annealing, and a final polish pass.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (150s limit, use 145s)
    global_deadline = time.time() + 145.0
    
    # Step 1: NEH heuristic
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    # Step 2: Fast adjacent swap local search with precomputed prefixes
    adj_time_limit = min(15.0, max(1.0, (global_deadline - time.time()) * 0.25))
    adj_perm, adj_ms = _fast_adjacent_swap_local_search(neh_perm, processing_times, time_limit=adj_time_limit)
    
    # Step 3: Neighborhood search (2-opt swaps)
    remaining_after_adj = global_deadline - time.time()
    ns_time_limit = min(25.0, max(1.0, remaining_after_adj * 0.25))
    ns_perm, ns_ms = _neighborhood_search(adj_perm, processing_times, time_limit=ns_time_limit)
    
    # Step 4: Simulated annealing if time remains
    remaining_time = global_deadline - time.time()
    if remaining_time > 10.0:
        sa_time_limit = min(remaining_time - 5.0, 40.0)
        sa_perm, sa_ms = _simulated_annealing(ns_perm, processing_times, time_limit=sa_time_limit)
        if sa_ms < ns_ms:
            ns_perm, ns_ms = sa_perm, sa_ms
    
    # Step 5: Final polish pass
    remaining_time = global_deadline - time.time()
    if remaining_time > 3.0:
        polish_time_limit = min(remaining_time - 2.0, 15.0)
        polish_perm, polish_ms = _polish_pass(ns_perm, processing_times, time_limit=polish_time_limit)
        if polish_ms < ns_ms:
            return polish_perm
    
    return ns_perm
# EVOLVE-BLOCK-END
