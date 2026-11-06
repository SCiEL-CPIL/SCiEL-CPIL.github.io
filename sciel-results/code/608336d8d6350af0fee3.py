"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])

def _compute_completion_matrix(perm, processing_times):
    """Compute the full completion time matrix C."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    if n_jobs == 0:
        return C
    
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        if j == 0:
            C[j, 0] = pt[0]
        else:
            C[j, 0] = C[j-1, 0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[m]
    return C

def _eval_swap_makespan(perm, C, i, processing_times):
    """
    Evaluate the makespan of swapping perm[i] and perm[i+1] given the current completion matrix C.
    Returns the new makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # Get previous row
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = C[i-1]
        
    job_a = perm[i]
    job_b = perm[i+1]
    pt_a = processing_times[job_a]
    pt_b = processing_times[job_b]
    
    # Compute new C[i] (job_b)
    C_new_i = np.empty(n_machines)
    C_new_i[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_new_i[m] = max(C_prev[m], C_new_i[m-1]) + pt_b[m]
        
    # Compute new C[i+1] (job_a)
    C_new_i1 = np.empty(n_machines)
    C_new_i1[0] = C_new_i[0] + pt_a[0]
    for m in range(1, n_machines):
        C_new_i1[m] = max(C_new_i[m], C_new_i1[m-1]) + pt_a[m]
        
    # Check if the new state is dominated by the old state
    # If C_new_i1 <= C[i+1] element-wise, the makespan will not increase.
    # But we need the exact new makespan.
    
    # If i+1 is the last job, we are done
    if i + 1 == n_jobs - 1:
        return float(C_new_i1[-1])
        
    # Propagate the change to the end
    C_curr = C_new_i1
    # We need to update C for the rest of the jobs
    # Note: C is the OLD matrix. We are computing the NEW makespan.
    # We don't modify C in-place here to keep it pure, but for efficiency in the caller, 
    # we might want to return the new tail or just the makespan.
    # For now, return makespan.
    
    for j in range(i + 2, n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.empty(n_machines)
        cur_row[0] = C_curr[0] + pt[0]
        for m in range(1, n_machines):
            cur_row[m] = max(C_curr[m], cur_row[m-1]) + pt[m]
        C_curr = cur_row
        
    return float(C_curr[-1])





def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    # Optimization: We can precompute the completion vector for the prefix of `partial`?
    # No, `partial` changes length as we insert.
    # But for a fixed `partial`, we can compute the completion vector for each prefix?
    # Let's just try all positions.
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _local_search(perm, processing_times, max_iterations=100, deadline=None):
    """
    Local search with 2-opt (adjacent swap) and insert moves.
    Uses incremental evaluation for swaps.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    C = _compute_completion_matrix(best_perm, processing_times)
    best_ms = float(C[-1, -1])
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adjacent swaps
        for i in range(n_jobs - 1):
            ms = _eval_swap_makespan(best_perm, C, i, processing_times)
            if ms < best_ms - 1e-10:
                # Apply the swap
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms
                # Update C only for the changed part?
                # For simplicity and safety, recompute C fully if we accept a move?
                # Or update C incrementally.
                # Recomputing C is O(N*M). Accepting a move is rare.
                # Let's recompute C to keep it consistent.
                C = _compute_completion_matrix(best_perm, processing_times)
                improved = True
                break
        
        if improved:
            continue
        
        # Try insert moves (Or-opt)
        # Move a job to any other position.
        if n_jobs <= 30:
            for i in range(n_jobs):
                job = best_perm[i]
                # Remove job
                perm_without = [p for p in best_perm if p != job]
                for pos in range(len(perm_without)):
                    candidate = perm_without[:pos] + [job] + perm_without[pos:]
                    ms = _makespan(candidate, processing_times)
                    if ms < best_ms - 1e-10:
                        best_perm = candidate
                        best_ms = ms
                        C = _compute_completion_matrix(best_perm, processing_times)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simulated annealing with adjacent swap and insert moves.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Temperature schedule
    # Start high, cool down.
    initial_temp = current_ms * 0.1
    final_temp = current_ms * 0.001
    # Number of steps estimate?
    # We don't know how many steps we can take.
    # Let's just use a geometric cooling factor.
    cooling_rate = 0.995
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < final_temp:
            break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = list(current_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = [p for p in current_perm if p != job]
            new_perm.insert(j, job)
            
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. LPT (Longest Processing Time)
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
        
    # 3. Randomized NEH
    if time.time() < deadline:
        rng2 = random.Random(rng.randint(0, 1000000))
        neh2_perm = _neh_order(processing_times, rng2)
        candidates.append(neh2_perm)
        
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
            
    # Local search polish
    if time.time() < deadline and n_jobs > 1:
        # Determine max iterations
        # NEH is O(N^2 * N*M) roughly. Local search is O(I * N * (N*M) + N^2 * (N*M)).
        # If N=50, M=5.
        # Let's be conservative.
        remaining = deadline - time.time()
        if remaining > 30:
             max_iter = 200
        elif remaining > 10:
             max_iter = 50
        else:
             max_iter = 10
             
        polished = _local_search(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
            
    # Simulated annealing if time remains
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
            
        # Final local search
        if time.time() < deadline - 2.0:
             final_polish = _local_search(best_perm, processing_times, max_iterations=50, deadline=deadline)
             final_ms = _makespan(final_polish, processing_times)
             if final_ms < best_ms:
                 best_perm = final_polish
                 best_ms = final_ms

    return list(best_perm)
# EVOLVE-BLOCK-END
