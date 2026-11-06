"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


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


def _compute_prefix_completions(perm, processing_times):
    """
    Compute completion time vectors for every prefix of the permutation.
    prefix_comps[i] is the completion vector after processing perm[0..i].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    prefix_comps = [None] * n_jobs
    if n_jobs == 0:
        return prefix_comps
    
    prev_row = np.zeros(n_machines)
    for i in range(n_jobs):
        job = perm[i]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
        prefix_comps[i] = prev_row.copy()
        
    return prefix_comps


def _eval_swap_adjacent(perm, i, processing_times, prefix_comps):
    """
    Evaluate makespan of swapping perm[i] and perm[i+1] using prefix completions.
    This is O(n*m) in the worst case but only recomputes the suffix.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = prefix_comps[i-1].copy()
    
    job_a = perm[i]
    job_b = perm[i+1]
    
    # Compute completion after job_b then job_a
    pt_b = processing_times[job_b]
    pt_a = processing_times[job_a]
    
    C_new = C_prev.copy()
    prev = 0.0
    for m in range(n_machines):
        new_time = max(C_new[m], prev) + pt_b[m]
        C_new[m] = new_time
        prev = new_time
    prev = 0.0
    for m in range(n_machines):
        new_time = max(C_new[m], prev) + pt_a[m]
        C_new[m] = new_time
        prev = new_time
    
    # Propagate the rest
    for k in range(i+2, n_jobs):
        job = perm[k]
        pt = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(C_new[m], prev) + pt[m]
            C_new[m] = new_time
            prev = new_time
            
    return C_new[-1]


def _eval_insert(perm, i, j, processing_times, prefix_comps):
    """
    Evaluate makespan of moving perm[i] to position j.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return _makespan(perm, processing_times)
    
    # Create a temporary list without the job at i
    # We need to be careful with indices
    if i < j:
        # Move job from i to j
        # Prefix up to i-1
        if i == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[i-1].copy()
            
        # Process jobs from i+1 to j
        C_mid = C_prev
        for k in range(i+1, j+1):
            job = perm[k]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(C_mid[m], prev) + pt[m]
                C_mid[m] = new_time
                prev = new_time
        
        # Insert job at i
        job_a = perm[i]
        pt_a = processing_times[job_a]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(C_mid[m], prev) + pt_a[m]
            C_mid[m] = new_time
            prev = new_time
            
        # Process rest
        for k in range(j+1, n_jobs):
            job = perm[k]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(C_mid[m], prev) + pt[m]
                C_mid[m] = new_time
                prev = new_time
                
        return C_mid[-1]
    else:
        # Move job from i to j (j < i)
        # Prefix up to j-1
        if j == 0:
            C_prev = np.zeros(n_machines)
        else:
            C_prev = prefix_comps[j-1].copy()
            
        # Insert job at j
        job_a = perm[i]
        pt_a = processing_times[job_a]
        prev = 0.0
        for m in range(n_machines):
            new_time = max(C_prev[m], prev) + pt_a[m]
            C_prev[m] = new_time
            prev = new_time
            
        # Process jobs from j to i-1
        for k in range(j, i):
            job = perm[k]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(C_prev[m], prev) + pt[m]
                C_prev[m] = new_time
                prev = new_time
                
        # Process rest
        for k in range(i+1, n_jobs):
            job = perm[k]
            pt = processing_times[job]
            prev = 0.0
            for m in range(n_machines):
                new_time = max(C_prev[m], prev) + pt[m]
                C_prev[m] = new_time
                prev = new_time
                
        return C_prev[-1]


def _neh_order(processing_times, rng=None):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling."""
    if rng is None:
        rng = random.Random()
        
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


def _tabu_search(perm, processing_times, time_limit=10.0, rng=None):
    """
    Tabu Search with 2-opt (swap) and insert moves.
    Uses a tabu list to prevent cycling.
    """
    if rng is None:
        rng = random.Random()
        
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    # Tabu list: stores (job, position) pairs that are tabu
    # Or better: stores (i, j) for swaps, and (i, j) for inserts
    # Let's use a dictionary: key=(job_id, pos), value=iteration when it becomes non-tabu
    tabu_list = {}
    tabu_tenure = max(5, n_jobs // 2)
    max_iterations = 20000
    iteration = 0
    start_time = time.time()
    
    # Precompute prefix completions for current perm
    prefix_comps = _compute_prefix_completions(current_perm, processing_times)
    
    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        iteration += 1
        
        # Clean up tabu list
        for key in list(tabu_list.keys()):
            if tabu_list[key] <= iteration:
                del tabu_list[key]
        
        best_move = None
        best_eval = float('inf')
        best_is_tabu = True
        
        # Generate moves
        # 1. Adjacent swaps
        for i in range(n_jobs - 1):
            # Check if swap (i, i+1) is tabu
            # We'll tabu the job moving to a specific position
            job_a = current_perm[i]
            job_b = current_perm[i+1]
            
            # Move job_a to i+1
            key_a = (job_a, i+1)
            is_tabu_a = key_a in tabu_list
            
            # Move job_b to i
            key_b = (job_b, i)
            is_tabu_b = key_b in tabu_list
            
            if not is_tabu_a and not is_tabu_b:
                is_tabu = False
            else:
                is_tabu = True
                
            # Evaluate swap
            # For speed, we can approximate or do full eval if not tabu
            # Let's do full eval for now, but only for non-tabu or if best_eval is inf
            # Actually, to be safe and fast, let's just evaluate a subset
            if iteration % 2 == 0 or i < n_jobs // 2: # Heuristic to limit evals
                eval_ms = _eval_swap_adjacent(current_perm, i, processing_times, prefix_comps)
                if eval_ms < best_eval:
                    best_eval = eval_ms
                    best_move = ('swap', i, None)
                    best_is_tabu = is_tabu
                    best_tabu_keys = [key_a, key_b]
        
        # 2. Insert moves (move job i to position j)
        # Limit number of insert moves evaluated
        if n_jobs > 5:
            for i in range(n_jobs):
                # Try a few random target positions
                for _ in range(2):
                    j = rng.randint(0, n_jobs - 1)
                    if i == j:
                        continue
                    
                    job_a = current_perm[i]
                    key = (job_a, j)
                    is_tabu = key in tabu_list
                    
                    if iteration % 3 == 0 or not is_tabu:
                        eval_ms = _eval_insert(current_perm, i, j, processing_times, prefix_comps)
                        if eval_ms < best_eval:
                            best_eval = eval_ms
                            best_move = ('insert', i, j)
                            best_is_tabu = is_tabu
                            best_tabu_keys = [key]
        
        if best_move is None:
            # No move found, restart or break
            break
        
        # Aspiration criterion: accept tabu move if it improves best solution
        if best_is_tabu and best_eval >= best_ms:
            # No non-tabu improving move, and tabu move doesn't improve best
            # Try to find any non-tabu move even if it worsens? Or break?
            # For now, let's just break if we can't find a good move
            # Or accept the best tabu move if it's better than current?
            if best_eval < current_ms:
                pass # Accept
            else:
                # Stagnation, maybe restart with new tabu list
                tabu_list = {}
                continue
        
        # Apply move
        move_type, i, j = best_move
        if move_type == 'swap':
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            # Add to tabu list
            for key in best_tabu_keys:
                if key is not None:
                    tabu_list[key] = iteration + tabu_tenure
        else: # insert
            job_a = current_perm[i]
            if i < j:
                current_perm.pop(i)
                current_perm.insert(j, job_a)
            else:
                current_perm.pop(i)
                current_perm.insert(j, job_a)
            for key in best_tabu_keys:
                if key is not None:
                    tabu_list[key] = iteration + tabu_tenure
        
        current_ms = best_eval
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = list(current_perm)
        
        # Update prefix completions
        prefix_comps = _compute_prefix_completions(current_perm, processing_times)
        
        # Periodically check time
        if time.time() - start_time > time_limit:
            break
            
    return best_perm, best_ms


def _simulated_annealing(perm, processing_times, time_limit=10.0, rng=None):
    """Simulated annealing for further improvement."""
    if rng is None:
        rng = random.Random()
        
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
        i = rng.randint(0, n_jobs - 1)
        j = rng.randint(0, n_jobs - 1)
        if i == j:
            continue
        
        candidate = list(current_perm)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_ms = _makespan(candidate, processing_times)
        
        delta = candidate_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
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
    Uses NEH heuristic followed by Tabu Search and Simulated Annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (150s limit, use 140s)
    global_deadline = time.time() + 140.0
    
    # Step 1: NEH heuristic
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    # Step 2: Tabu Search
    ts_time_limit = min(60.0, max(5.0, (global_deadline - time.time()) * 0.5))
    ts_perm, ts_ms = _tabu_search(neh_perm, processing_times, time_limit=ts_time_limit)
    
    # Step 3: Simulated Annealing if time remains
    remaining_time = global_deadline - time.time()
    if remaining_time > 2.0:
        sa_time_limit = min(remaining_time - 1.0, 30.0)
        sa_perm, sa_ms = _simulated_annealing(ts_perm, processing_times, time_limit=sa_time_limit)
        if sa_ms < ts_ms:
            return sa_perm
    
    return ts_perm
# EVOLVE-BLOCK-END
