"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation efficiently."""
    n = len(perm)
    m = pt.shape[1]
    # completion times: C[j][k] for job j on machine k
    # We can compute this incrementally
    # For a flow shop, the completion time of job i on machine k is:
    # C[i][k] = max(C[i-1][k], C[i][k-1]) + p[i][k]
    # where C[i-1][k] is completion of previous job on machine k
    # and C[i][k-1] is completion of same job on previous machine
    
    # Use vectorized approach: compute the flow shop completion times
    # For each job in order, track the completion time on each machine
    prev_machine_time = np.zeros(m)  # completion time of previous job on each machine
    makespan = 0.0
    for j in perm:
        curr_time = 0.0
        for k in range(m):
            curr_time = max(curr_time, prev_machine_time[k]) + pt[j, k]
            prev_machine_time[k] = curr_time
        makespan = curr_time
    return makespan


def _makespan_fast(perm, pt):
    """Faster makespan computation using numpy."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    
    # Extract processing times in permutation order
    p_ordered = pt[perm]  # shape (n, m)
    
    # Compute completion times using the recurrence:
    # C[i][k] = max(C[i-1][k], C[i][k-1]) + p[i][k]
    # This is a 2D DP that can be done iteratively
    
    # For efficiency, we'll do it with a loop over jobs but vectorize over machines
    prev_row = np.zeros(m)  # C[i-1][k] for all k
    for i in range(n):
        curr_row = np.zeros(m)
        for k in range(m):
            curr_row[k] = max(prev_row[k], curr_row[k-1] if k > 0 else 0.0) + p_ordered[i, k]
        prev_row = curr_row
    
    return prev_row[-1]


def _neh_construct(pt):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = pt.shape
    
    # Sort jobs by decreasing total processing time
    total_times = pt.sum(axis=1)
    order = list(np.argsort(-total_times))
    
    seq = [order[0]]
    for j in order[1:]:
        best_pos = 0
        best_makespan = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [j] + seq[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        seq = seq[:best_pos] + [j] + seq[best_pos:]
    
    return seq


def _insertion_local_search(perm, pt, max_passes=10):
    """Local search with insertion moves."""
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    current_makespan = _makespan(perm, pt)
    best_perm = list(perm)
    best_makespan = current_makespan
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            job = best_perm[i]
            # Remove job from position i
            remaining = best_perm[:i] + best_perm[i+1:]
            # Try inserting at each position
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan(candidate, pt)
                if ms < best_makespan - 1e-10:
                    best_makespan = ms
                    best_perm = candidate
                    improved = True
                    # Restart the pass from the beginning since permutation changed
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_perm, best_makespan


def _perturb(perm, pt, strength=3):
    """Perturb the permutation by reversing a random segment."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    perm = list(perm)
    # Choose a random segment to reverse
    for _ in range(strength):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i > j:
            i, j = j, i
        if j - i >= 1:
            perm[i:j+1] = perm[i:j+1][::-1]
    
    return perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using:
    1. NEH heuristic for initial construction
    2. Iterated local search with insertion moves and perturbation
    """
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard: 120s limit, use 110s as safety margin
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial solution via NEH
    best_perm = _neh_construct(pt)
    best_makespan = _makespan(best_perm, pt)
    
    # Iterated local search
    current_perm = list(best_perm)
    current_makespan = best_makespan
    
    np.random.seed(42)  # For reproducibility
    
    while time.time() < deadline:
        # Local search on current permutation
        ls_perm, ls_makespan = _insertion_local_search(current_perm, pt, max_passes=5)
        
        if ls_makespan < best_makespan - 1e-10:
            best_perm = list(ls_perm)
            best_makespan = ls_makespan
            current_perm = list(ls_perm)
            current_makespan = ls_makespan
        else:
            # Perturb and continue
            current_perm = _perturb(current_perm, pt, strength=2)
            # Quick local search on perturbed solution
            current_perm, current_makespan = _insertion_local_search(current_perm, pt, max_passes=3)
            if current_makespan < best_makespan - 1e-10:
                best_perm = list(current_perm)
                best_makespan = current_makespan
    
    return list(best_perm)
# EVOLVE-BLOCK-END
