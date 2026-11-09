# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, perm):
    """Compute makespan efficiently."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + P[perm[j]][k]
    # We only need the last row for the makespan, but we need the previous row to compute it.
    # We can do this in O(n*m) time and O(m) space.
    
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    P = processing_times
    
    for j in range(n):
        job = perm[j]
        if j == 0:
            curr_row[0] = P[job, 0]
            for k in range(1, m):
                curr_row[k] = curr_row[k-1] + P[job, k]
        else:
            curr_row[0] = prev_row[0] + P[job, 0]
            for k in range(1, m):
                # max(curr_row[k-1], prev_row[k]) + P[job, k]
                val = P[job, k]
                cm1 = curr_row[k-1]
                pk = prev_row[k]
                curr_row[k] = (cm1 if cm1 > pk else pk) + val
        
        # Swap rows
        prev_row, curr_row = curr_row, prev_row
        
    return prev_row[-1]

def _c_ds(processing_times):
    """Composite Dispatching Heuristic."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    candidates = []
    
    # SPT and LPT on each machine, plus reverse
    for k in range(m):
        p_k = processing_times[:, k]
        # SPT
        spt = np.argsort(p_k)
        candidates.append(list(spt))
        candidates.append(list(spt[::-1]))
        # LPT
        lpt = np.argsort(-p_k)
        candidates.append(list(lpt))
        candidates.append(list(lpt[::-1]))
        
    # Also try sorting by total processing time
    totals = processing_times.sum(axis=1)
    spt_total = np.argsort(totals)
    candidates.append(list(spt_total))
    candidates.append(list(spt_total[::-1]))
    lpt_total = np.argsort(-totals)
    candidates.append(list(lpt_total))
    candidates.append(list(lpt_total[::-1]))
    
    best_perm = candidates[0]
    best_ms = _makespan(processing_times, best_perm)
    
    for perm in candidates[1:]:
        ms = _makespan(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
            
    return best_perm

def _simulated_annealing(processing_times, initial_perm, deadline):
    """Simulated Annealing with Swap and Block Reversal moves."""
    n = len(initial_perm)
    if n < 2:
        return list(initial_perm), _makespan(processing_times, initial_perm)
    
    rng = np.random.default_rng(42)
    
    current_perm = list(initial_perm)
    current_ms = _makespan(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # SA Parameters
    T0 = current_ms * 0.1 if current_ms > 0 else 1.0
    T_min = 1e-6
    alpha = 0.95 # Cooling rate
    
    T = T0
    iterations = 0
    max_iters = 100000 # Hard cap, but time is the real limit
    
    while time.time() < deadline and T > T_min and iterations < max_iters:
        # Generate a neighbor
        move_type = rng.integers(0, 3) # 0: Swap, 1: Insert, 2: Block Reverse
        
        new_perm = current_perm[:]
        
        if move_type == 0:
            # Swap
            i, j = rng.choice(n, 2, replace=False)
            new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        elif move_type == 1:
            # Insert
            i = rng.integers(0, n)
            job = new_perm.pop(i)
            j = rng.integers(0, n)
            new_perm.insert(j, job)
        else:
            # Block Reverse (2-opt)
            i, j = rng.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            new_perm[i:j+1] = new_perm[i:j+1][::-1]
            
        new_ms = _makespan(processing_times, new_perm)
        delta = new_ms - current_ms
        
        # Acceptance criterion
        if delta < 0:
            # Better, accept
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        else:
            # Worse, accept with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
        
        # Cool down
        T *= alpha
        iterations += 1
        
    return best_perm, best_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    # Safety margin: 140s out of 150s limit
    deadline = start_time + 140
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 0:
        return list(range(n))
    
    # 1. Generate initial solution using CDS
    best_perm = _c_ds(processing_times)
    best_ms = _makespan(processing_times, best_perm)
    
    # 2. Check time
    if time.time() > deadline:
        return best_perm
        
    # 3. Run Simulated Annealing
    # We might run SA multiple times if time permits, starting from best found
    # But typically one long run is better for SA.
    # Let's give SA most of the time.
    sa_deadline = deadline - 5 # Leave 5s for cleanup/return
    
    sa_perm, sa_ms = _simulated_annealing(processing_times, best_perm, sa_deadline)
    
    if sa_ms < best_ms:
        best_perm = sa_perm
        best_ms = sa_ms
        
    return best_perm
# EVOLVE-BLOCK-END
