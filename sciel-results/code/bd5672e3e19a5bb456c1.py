"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_completion(perm, pt):
    """Compute completion times matrix [n_machines, n_jobs] for a permutation."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    # C[m, j] = completion time of job j on machine m
    C = np.zeros((n_machines, n_jobs))
    for j_idx, job in enumerate(perm):
        for m in range(n_machines):
            if m == 0:
                C[m, j_idx] = pt[job, 0]
            else:
                C[m, j_idx] = max(C[m-1, j_idx], C[m, j_idx-1]) + pt[job, m]
    return C


def _makespan_from_C(C):
    """Makespan is the max completion on the last machine."""
    return C[-1, -1]


def _delta_insert(perm, C, pt, pos, job):
    """
    Compute the change in makespan when inserting `job` at position `pos` 
    into the current permutation `perm` (removing it from its current position).
    Returns (delta_makespan, new_perm_list) for the candidate.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Find current position of job
    cur_pos = perm.index(job)
    
    # If job is already at pos, no change
    if cur_pos == pos:
        return 0.0, list(perm)
    
    # Build new permutation
    new_perm = list(perm)
    del new_perm[cur_pos]
    new_perm.insert(pos, job)
    
    # Compute new makespan using the completion matrix update
    # This is O(n*m) but we can vectorize the forward pass
    # For efficiency in SA, we compute full makespan but with numpy
    # Actually, let's just compute the new makespan directly
    # To keep it fast, we'll use a simplified incremental approach for the last machine
    # But for correctness, let's compute the new C matrix incrementally from min(cur_pos, pos)
    
    # Determine the earliest index where the schedule changes
    if cur_pos < pos:
        # Job moved right: indices from cur_pos to pos-1 shift left, job inserted at pos
        # Changes start at cur_pos
        start_idx = cur_pos
    else:
        # Job moved left: indices from pos to cur_pos-1 shift right, job inserted at pos
        start_idx = pos
    
    # Recompute C from start_idx onwards
    new_C = C.copy()
    for j_idx in range(start_idx, n_jobs):
        job_id = new_perm[j_idx]
        for m in range(n_machines):
            if m == 0:
                new_C[m, j_idx] = pt[job_id, 0]
            else:
                prev_machine = new_C[m-1, j_idx]
                prev_job = new_C[m, j_idx-1] if j_idx > 0 else 0
                new_C[m, j_idx] = max(prev_machine, prev_job) + pt[job_id, m]
    
    old_mk = C[-1, -1]
    new_mk = new_C[-1, -1]
    return new_mk - old_mk, new_perm


def _sa_search(perm, pt, start_time, deadline):
    """Simulated Annealing with insert moves."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    C = _compute_completion(perm, pt)
    current_mk = _makespan_from_C(C)
    best_perm = list(perm)
    best_mk = current_mk
    
    # Cooling schedule
    T_start = current_mk * 0.1 + 1.0
    T_end = 1e-6
    # Aim for ~5000-10000 iterations depending on size
    target_iters = min(10000, max(2000, n_jobs * 100))
    cooling_rate = (T_end / T_start) ** (1.0 / target_iters) if target_iters > 0 else 1.0
    T = T_start
    
    iters = 0
    while iters < target_iters:
        if time.time() > deadline:
            break
        
        # Random insert move
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j or i == j + 1:
            iters += 1
            T *= cooling_rate
            continue
        
        # Move job at position i to position j (adjusting j if i < j)
        job = perm[i]
        # Remove from i
        perm_no_job = perm[:i] + perm[i+1:]
        # Insert at j (if j > i, the position in perm_no_job is j-1)
        insert_pos = j if j > i else j
        if insert_pos >= len(perm_no_job):
            insert_pos = len(perm_no_job) - 1
        
        new_perm = perm_no_job[:insert_pos] + [job] + perm_no_job[insert_pos:]
        
        # Compute new makespan
        # Incremental update
        if i < insert_pos:
            start_idx = i
        else:
            start_idx = insert_pos
        
        new_C = C.copy()
        for j_idx in range(start_idx, n_jobs):
            job_id = new_perm[j_idx]
            for m in range(n_machines):
                if m == 0:
                    new_C[m, j_idx] = pt[job_id, 0]
                else:
                    prev_m = new_C[m-1, j_idx]
                    prev_j = new_C[m, j_idx-1] if j_idx > 0 else 0
                    new_C[m, j_idx] = max(prev_m, prev_j) + pt[job_id, m]
        
        new_mk = new_C[-1, -1]
        delta = new_mk - current_mk
        
        if delta < 0:
            # Accept improvement
            perm = new_perm
            C = new_C
            current_mk = new_mk
            if current_mk < best_mk:
                best_mk = current_mk
                best_perm = list(perm)
        else:
            # Accept with probability
            if T > T_end:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    perm = new_perm
                    C = new_C
                    current_mk = new_mk
        
        iters += 1
        T *= cooling_rate
    
    return best_perm, best_mk


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH-like heuristic (fast version)
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    init_order = list(np.argsort(-totals))
    
    # Run multiple SA restarts with different initial solutions
    best_perm = init_order
    best_mk = float('inf')
    
    # Create diverse initial solutions
    initial_solutions = [init_order]
    
    # SPT order
    spt_order = list(np.argsort(totals))
    initial_solutions.append(spt_order)
    
    # Random orders
    random.seed(123)
    for _ in range(3):
        o = list(range(n_jobs))
        random.shuffle(o)
        initial_solutions.append(o)
    
    for init in initial_solutions:
        if time.time() > deadline:
            break
        
        sa_perm, sa_mk = _sa_search(init, processing_times, start_time, deadline)
        
        if sa_mk < best_mk:
            best_mk = sa_mk
            best_perm = sa_perm
    
    # Final local search polish on the best solution
    if time.time() < deadline - 5.0:
        # Simple 2-opt polish
        n = len(best_perm)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, min(i + 10, n)):  # Limited window
                    new_perm = list(best_perm)
                    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                    C = _compute_completion(new_perm, processing_times)
                    mk = _makespan_from_C(C)
                    if mk < best_mk - 1e-9:
                        best_perm = new_perm
                        best_mk = mk
                        improved = True
    
    return best_perm
# EVOLVE-BLOCK-END
