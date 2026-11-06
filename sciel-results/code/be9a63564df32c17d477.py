"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(perm) == 0:
        return 0.0
    completion = [0.0] * n_machines
    
    for job in perm:
        row = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]


def _calc_makespan_with_prefix(processing_times, perm, prefix_len):
    """
    Calculate makespan, returning both the makespan and completion array.
    If prefix_len is given, only recompute from prefix_len onward (assumes
    completion state at prefix_len is correct for the current perm).
    Returns (makespan, completion_list).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(perm) == 0:
        return 0.0, [0.0] * n_machines
    
    if prefix_len is None or prefix_len <= 0:
        completion = [0.0] * n_machines
        start = 0
    else:
        # Recompute from scratch up to prefix_len to get correct state
        completion = [0.0] * n_machines
        for job in perm[:prefix_len]:
            row = processing_times[job]
            for m in range(n_machines):
                if m == 0:
                    completion[0] += row[m]
                else:
                    c_prev = completion[m-1]
                    c_curr = completion[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    completion[m] = c_curr + row[m]
        start = prefix_len
    
    for idx in range(start, len(perm)):
        row = processing_times[perm[idx]]
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1], completion


def _neh_heuristic(processing_times, seed=None):
    """
    NEH heuristic with incremental evaluation for insertion.
    If seed is provided, uses it for randomization of tie-breaking or initial ordering perturbation.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    
    if seed is not None:
        rng = random.Random(seed)
        # Add small random perturbation to break ties and create diversity
        perturbed_totals = np.array([t + rng.uniform(0, 0.01 * (totals.max() - totals.min() + 1e-9)) for t in totals])
        order = np.argsort(-perturbed_totals)
    else:
        order = np.argsort(-totals)
    
    sequence = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            # Build candidate sequence
            if pos == 0:
                cand = [job] + sequence
            elif pos == len(sequence):
                cand = sequence + [job]
            else:
                cand = sequence[:pos] + [job] + sequence[pos:]
            
            ms = _calc_makespan(processing_times, cand)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
    
    return sequence


def _local_search(processing_times, sequence, deadline, max_radius=1):
    """
    Insertion local search with incremental makespan evaluation.
    If max_radius > 1, performs Variable Neighborhood Search (VNS) by increasing radius.
    """
    n_jobs = len(sequence)
    if n_jobs <= 1:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    # If max_radius is 1, we just do standard insertion search (radius 1 means adjacent swaps essentially, 
    # but insertion search covers all moves. Standard insertion LS is effectively radius N, 
    # but VNS structure implies we start with small radius and expand).
    
    # The prompt suggests: "performing random insertions of jobs from position i to j with |i-j| <= k, incrementing k"
    # However, standard insertion local search (move any job to any position) is a superset of radius-k moves.
    # To implement VNS properly:
    # 1. Start with k=1 (adjacent swaps or short moves)
    # 2. If no improvement, increase k.
    # 3. If improvement found, reset k to 1.
    
    # For efficiency, we will implement a standard insertion LS but structured to allow early termination 
    # or specific radius constraints if desired. Given the time budget, full insertion LS is usually best.
    # But to follow the directive "Variable Neighborhood Search ... incrementing k until a local optimum is reached",
    # we will implement a loop that increases the maximum allowed distance for a move.
    
    # Note: Full insertion search is equivalent to k = n_jobs.
    # If we restrict k, we might miss better moves.
    # Strategy:
    # - Use full insertion search (k = n_jobs) as the base.
    # - To escape local minima, we can use a "shake" (random perturbation) if stuck.
    
    # Let's implement a hybrid:
    # 1. Standard insertion LS (full radius).
    # 2. If stuck (no improvement in a full pass), perform a random perturbation (shake) and restart.
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    k = 1
    max_k = n_jobs - 1
    
    while time.time() < deadline:
        improved = False
        # Explore neighborhood with current radius k
        # For efficiency, if k is small, we only check moves where |i-j| <= k.
        # If k is large, we check all.
        
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            # Determine range of j to check based on radius k
            j_min = max(0, i - k)
            j_max = min(n_jobs - 1, i + k)
            
            for j in range(j_min, j_max + 1):
                if j == i or j == i + 1:
                    continue
                if time.time() >= deadline:
                    break
                
                # Remove job at i, insert at position j (adjusted)
                if j > i:
                    insert_idx = j - 1
                else:
                    insert_idx = j
                
                # Build new sequence
                seq_copy = current_seq[:]
                seq_copy.pop(i)
                seq_copy.insert(insert_idx, job)
                
                # Incremental evaluation
                if j > i:
                    prefix_len = i
                else:
                    prefix_len = insert_idx
                
                new_makespan, _ = _calc_makespan_with_prefix(
                    processing_times, seq_copy, prefix_len
                )
                
                if new_makespan < current_makespan - 1e-9:
                    current_seq = seq_copy
                    current_makespan = new_makespan
                    improved = True
                    
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_seq = list(current_seq)
                    
                    # Reset radius to 1 after an improvement (VNS strategy)
                    k = 1
                    break
            
            if improved:
                break
        
        if not improved:
            # No improvement found with current radius
            if k < max_k:
                k += 1
            else:
                # Reached maximum radius (full neighborhood) and no improvement
                # Perform a shake (random perturbation) to escape local minimum
                if time.time() >= deadline - 1.0:
                    break
                
                # Random perturbation: swap two random jobs
                idx1, idx2 = random.sample(range(n_jobs), 2)
                current_seq[idx1], current_seq[idx2] = current_seq[idx2], current_seq[idx1]
                current_makespan = _calc_makespan(processing_times, current_seq)
                
                # Reset radius
                k = 1
    
    return best_seq, best_makespan


def _simulated_annealing(processing_times, sequence, deadline):
    """
    Simulated annealing to escape local minima.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    T = current_makespan * 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if n_jobs < 2:
            break
        i, j = random.sample(range(n_jobs), 2)
        new_seq = current_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        
        new_makespan = _calc_makespan(processing_times, new_seq)
        delta = new_makespan - current_makespan
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq = new_seq
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_seq = list(current_seq)
        
        T *= cooling_rate
        if T < 1e-6:
            break
    
    return best_seq, best_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    
    # Step 1: Generate multiple diverse initial solutions using NEH with randomization
    # We will use a few restarts to find a good starting point.
    best_seq = None
    best_makespan = float('inf')
    
    # Allocate some time for initialization (e.g., 5 seconds)
    init_deadline = start_time + 5.0
    
    for seed in range(10):
        if time.time() >= init_deadline:
            break
        
        seq = _neh_heuristic(processing_times, seed=seed)
        ms = _calc_makespan(processing_times, seq)
        
        if ms < best_makespan:
            best_makespan = ms
            best_seq = seq
    
    if best_seq is None:
        # Fallback if loop didn't run
        best_seq = _neh_heuristic(processing_times)
        best_makespan = _calc_makespan(processing_times, best_seq)
    
    # Step 2: VNS Local Search with time limit
    # Use the VNS approach to refine the best initial solution
    search_deadline = deadline - 10.0  # Leave 10s for final polish/SA
    vns_seq, vns_makespan = _local_search(processing_times, best_seq, search_deadline)
    
    if vns_makespan < best_makespan:
        best_seq = vns_seq
        best_makespan = vns_makespan
    
    # Step 3: Simulated Annealing if time permits
    if time.time() < deadline - 5.0:
        sa_seq, sa_makespan = _simulated_annealing(processing_times, best_seq, deadline - 5.0)
        if sa_makespan < best_makespan:
            best_seq = sa_seq
            best_makespan = sa_makespan
    
    # Step 4: Final local search polish
    if time.time() < deadline - 2.0:
        final_seq, final_makespan = _local_search(processing_times, best_seq, deadline)
        if final_makespan < best_makespan:
            best_seq = final_seq
    
    return best_seq


# EVOLVE-BLOCK-END
