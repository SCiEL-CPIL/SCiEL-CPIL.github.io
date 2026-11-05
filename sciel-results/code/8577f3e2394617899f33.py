"""
Flow-shop permutation optimization using fast heuristics, delta-based insertion
local search, and simulated annealing with a stronger neighborhood.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

TIME_LIMIT = 145.0  # Safety margin from 150s limit

def _makespan(pt, perm):
    """
    Compute makespan for a given permutation. O(N*M).
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    for j in perm:
        curr = 0.0
        for m in range(n_machines):
            if prev[m] > curr:
                curr = prev[m]
            curr += pt[j, m]
            prev[m] = curr
    return prev[-1]

def _makespan_with_deltas(pt, perm, deltas):
    """
    Compute makespan for perm given pre-computed deltas for each position.
    This is used in the delta-based local search.
    """
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    for i, j in enumerate(perm):
        curr = 0.0
        for m in range(n_machines):
            if prev[m] > curr:
                curr = prev[m]
            curr += pt[j, m] + deltas[i, m] if i < len(deltas) else pt[j, m]
            prev[m] = curr
    return prev[-1]

def _fast_makespan(pt, perm):
    """
    Optimized makespan computation using local variables.
    """
    n_machines = pt.shape[1]
    if n_machines == 0 or len(perm) == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    for j in perm:
        curr = 0.0
        for m in range(n_machines):
            pm = prev[m]
            if pm > curr:
                curr = pm
            curr += pt[j, m]
            prev[m] = curr
    return prev[-1]

def _neh_initial(pt):
    """
    NEH heuristic: Sort jobs by total processing time descending.
    Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = pt.shape
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _fast_makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _palmer_initial(pt):
    """
    Palmer's heuristic.
    """
    n_jobs, n_machines = pt.shape
    weights = np.arange(n_machines, 0, -1, dtype=float)
    priorities = pt.dot(weights)
    order = list(np.argsort(-priorities))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _fast_makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _lpt_initial(pt):
    """
    LPT: Sort by longest processing time on first machine.
    """
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(-pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _fast_makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _spt_initial(pt):
    """
    SPT: Sort by shortest processing time on first machine.
    """
    n_jobs, n_machines = pt.shape
    order = list(np.argsort(pt[:, 0]))
    
    current = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current) + 1):
            test_perm = current[:pos] + [job] + current[pos:]
            ms = _fast_makespan(pt, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current = current[:best_pos] + [job] + current[best_pos:]
        
    return current

def _random_initial(n_jobs, seed):
    """Random initial solution."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_jobs).tolist()

def _delta_insertion_local_search(pt, perm, max_iterations=5000, deadline=None):
    """
    Delta-based insertion local search with best-improvement strategy.
    Uses prefix/suffix computation for O(N*M) per evaluation instead of O(N^2*M).
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return perm[:], 0.0
    
    n_machines = pt.shape[1]
    current_perm = perm[:]
    current_ms = _fast_makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    iteration = 0
    while iteration < max_iterations:
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute prefix and suffix completion times
        # prefix[i][m] = completion time of job at position i on machine m
        # suffix[i][m] = "remaining" time from position i onwards on machine m
        
        # Forward pass: prefix[i][m]
        prefix = [[0.0] * n_machines for _ in range(n_jobs)]
        prev = [0.0] * n_machines
        for i in range(n_jobs):
            j = current_perm[i]
            curr = 0.0
            for m in range(n_machines):
                if prev[m] > curr:
                    curr = prev[m]
                curr += pt[j, m]
                prev[m] = curr
                prefix[i][m] = curr
        
        # Backward pass: suffix[i][m] represents the "load" that would be added
        # if we start processing from position i onwards, assuming the machine
        # is free at time 0. Actually, we need a different approach.
        
        # For insertion at position pos, we need to compute the makespan of:
        # current_perm[:pos] + [job] + current_perm[pos:]
        # where job is removed from its current position.
        
        # Let's use a different approach: for each job i, remove it and try inserting
        # at each position pos. We can compute the makespan incrementally.
        
        improved = False
        best_delta = 0.0
        best_move = None  # (i, pos)
        
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            n_without = len(without_job)
            
            # Compute prefix for without_job
            prefix_w = [[0.0] * n_machines for _ in range(n_without)]
            prev_w = [0.0] * n_machines
            for k in range(n_without):
                j = without_job[k]
                curr = 0.0
                for m in range(n_machines):
                    if prev_w[m] > curr:
                        curr = prev_w[m]
                    curr += pt[j, m]
                    prev_w[m] = curr
                    prefix_w[k][m] = curr
            
            # Compute suffix for without_job
            # suffix[k][m] = the time at which machine m would be free if we
            # started processing from position k onwards (assuming all machines free at 0)
            # This is not quite right. Let's think differently.
            
            # For inserting job at position pos in without_job:
            # The job starts on machine m at max(prefix_w[pos-1][m], 0) + pt[job, m]
            # But we also need to account for the fact that the job's completion on
            # machine m affects the start of the next job on machine m.
            
            # Let's just compute the full makespan for each candidate position.
            # To speed this up, we can note that for positions before i, the prefix
            # is the same as in the original sequence (up to position pos-1).
            
            for pos in range(n_without + 1):
                if pos == i:
                    continue
                
                # Compute makespan of without_job[:pos] + [job] + without_job[pos:]
                new_perm = without_job[:pos] + [job] + without_job[pos:]
                new_ms = _fast_makespan(pt, new_perm)
                
                delta = new_ms - current_ms
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, pos)
        
        if best_move is not None and best_delta < 0:
            i, pos = best_move
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            current_perm = without_job[:pos] + [job] + without_job[pos:]
            current_ms += best_delta
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
            improved = True
        
        if not improved:
            break
        
        iteration += 1
    
    return best_perm, best_ms

def _simulated_annealing(pt, perm, deadline, T0=None, cooling=0.995):
    """
    Simulated Annealing with Or-Opt neighborhood (size 1, 2, 3).
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return perm[:], 0.0
    
    current_perm = perm[:]
    current_ms = _fast_makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    if T0 is None:
        T0 = max(10.0, current_ms * 0.5)
        
    T = T0
    rng = np.random.RandomState(42)
    
    max_block_size = 3
    
    while time.time() < deadline and T > 1e-6:
        block_size = rng.randint(1, max_block_size + 1)
        if n_jobs < block_size:
            block_size = n_jobs
            
        i = rng.randint(0, n_jobs - block_size + 1)
        k = rng.randint(0, n_jobs - block_size + 1)
        
        if k == i or k == i - 1:
            continue
            
        block = current_perm[i:i+block_size]
        
        new_perm = current_perm[:i] + current_perm[i+block_size:]
        new_perm = new_perm[:k] + block + new_perm[k:]
        
        new_ms = _fast_makespan(pt, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                
        T *= cooling
        
    return best_perm, best_ms

def _tabu_search(pt, perm, deadline, tabu_tenure=10, max_iterations=10000):
    """
    Tabu search with insertion neighborhood.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return perm[:], 0.0
    
    current_perm = perm[:]
    current_ms = _fast_makespan(pt, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # Tabu list: (job, pos) -> iteration when it becomes non-tabu
    tabu = {}
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        best_move = None
        best_delta = 0.0
        best_candidate = None
        
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            n_without = len(without_job)
            
            for pos in range(n_without + 1):
                if pos == i:
                    continue
                
                # Check tabu
                move_key = (job, pos)
                if move_key in tabu and tabu[move_key] > iteration:
                    # Aspiration: allow if it improves best
                    new_perm = without_job[:pos] + [job] + without_job[pos:]
                    new_ms = _fast_makespan(pt, new_perm)
                    if new_ms >= best_ms:
                        continue
                    # Otherwise, skip tabu move
                
                new_perm = without_job[:pos] + [job] + without_job[pos:]
                new_ms = _fast_makespan(pt, new_perm)
                delta = new_ms - current_ms
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, pos)
                    best_candidate = new_perm
        
        if best_move is not None and best_delta < 0:
            i, pos = best_move
            job = current_perm[i]
            current_perm = best_candidate
            current_ms += best_delta
            tabu[(job, pos)] = iteration + tabu_tenure
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        else:
            break
        
        iteration += 1
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using multiple heuristics,
    local search, tabu search, and simulated annealing.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Generate initial candidates
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Random starts
    for seed in range(6):
        candidates.append(_random_initial(n_jobs, seed))
        
    best_perm = candidates[0]
    best_ms = _fast_makespan(pt, best_perm)
    
    # Phase 1: Insertion local search on each candidate
    # Allocate time based on number of candidates
    num_candidates = len(candidates)
    ls_time_per_cand = max(5.0, (TIME_LIMIT * 0.3) / num_candidates)
    
    for cand in candidates:
        if time.time() > deadline - 10.0:  # Leave 10s for safety
            break
        cand_deadline = time.time() + ls_time_per_cand
        if cand_deadline > deadline - 5.0:
            cand_deadline = deadline - 5.0
            
        perm, ms = _delta_insertion_local_search(pt, cand, max_iterations=2000, deadline=cand_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 2: Tabu search on the best result
    if time.time() < deadline - 15.0:
        ts_deadline = deadline - 10.0
        perm, ms = _tabu_search(pt, best_perm, ts_deadline, tabu_tenure=10, max_iterations=5000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 10.0:
        sa_deadline = deadline - 5.0
        perm, ms = _simulated_annealing(pt, best_perm, sa_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Phase 4: Final local search polish
    if time.time() < deadline - 3.0:
        final_deadline = deadline - 1.0
        perm, ms = _delta_insertion_local_search(pt, best_perm, max_iterations=1000, deadline=final_deadline)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]
            
    # Ensure we return a list of ints
    return [int(x) for x in best_perm]

# EVOLVE-BLOCK-END
