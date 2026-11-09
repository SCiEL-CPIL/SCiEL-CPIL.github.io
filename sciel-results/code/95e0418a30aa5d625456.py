"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation on the flow shop."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # Cumulative completion times on each machine
    # machine_time[machine] tracks when that machine becomes free
    machine_time = np.zeros(m)
    for j in perm:
        for k in range(m):
            start = max(machine_time[k], machine_time[k-1]) if k > 0 else machine_time[k]
            # Actually we need to track per-machine completion properly
            # Let's use a 2D approach or iterative
            pass
    
    # More efficient: use the standard flow shop makespan calculation
    # completion[j][k] = max(completion[j][k-1], completion[j-1][k]) + p[j][k]
    # But we want to avoid allocating large arrays if possible
    
    # For small n this is fine
    n_jobs = n
    n_machines = m
    
    # completion[j][k] for job index j in permutation order, machine k
    # We only need the previous row to compute the current row
    prev_row = np.zeros(n_machines)
    current_row = np.zeros(n_machines)
    
    for j in range(n_jobs):
        job_idx = perm[j]
        current_row[0] = prev_row[0] + pt[job_idx, 0]
        for k in range(1, n_machines):
            current_row[k] = max(prev_row[k], current_row[k-1]) + pt[job_idx, k]
        prev_row = current_row.copy()
    
    return prev_row[-1]


def _makespan_fast(perm, pt):
    """Faster makespan using numpy where possible."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0.0
    if m == 1:
        return pt[perm, 0].sum()
    
    # Build the submatrix for the permutation
    sub = pt[perm, :]  # shape (n, m)
    
    # completion[i][j] = max(completion[i-1][j], completion[i][j-1]) + sub[i][j]
    # We can do this iteratively
    comp = np.zeros((n, m))
    comp[0, 0] = sub[0, 0]
    for j in range(1, m):
        comp[0, j] = comp[0, j-1] + sub[0, j]
    for i in range(1, n):
        comp[i, 0] = comp[i-1, 0] + sub[i, 0]
        for j in range(1, m):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + sub[i, j]
    
    return comp[-1, -1]


def _neh_order(pt):
    """NEH heuristic: order jobs by total processing time descending, 
    inserting each job in the best position."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with first job
    perm = [order[0]]
    
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm


def _local_search(perm, pt, time_limit=10.0):
    """2-opt style local search: try swapping adjacent jobs."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n - 1):
            if (time.time() - start_time) >= time_limit:
                break
            # Try swapping jobs at positions i and i+1
            new_perm = best_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            ms = _makespan_fast(new_perm, pt)
            if ms < best_ms - 1e-9:
                best_perm = new_perm
                best_ms = ms
                improved = True
    
    return best_perm


def _simulated_annealing(perm, pt, time_limit=5.0, n_restarts=3):
    """Simulated annealing with 2-opt moves (swap two random positions)."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    start_time = time.time()
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.95
    
    rng = np.random.default_rng(42)
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Pick two random positions to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan_fast(new_perm, pt)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        T *= cooling
    
    return best_perm


def _palmer_key(pt):
    """Compute Palmer (WSPT-like) key for each job."""
    n = pt.shape[0]
    m = pt.shape[1]
    keys = np.zeros(n)
    for j in range(n):
        # Palmer key: (n - 2k - 1) * p[j][k] summed over machines
        # Actually Palmer: sum over k of (n - 2k - 1) * p[j][k] where k is machine index (0-based)
        # More standard: key_j = sum_{k=0}^{m-1} (n - 2k - 1) * p[j][k]
        for k in range(m):
            keys[j] += (n - 2*k - 1) * pt[j, k]
    return keys


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    if m == 1:
        # Single machine: SPT (shortest processing time first)
        return list(np.argsort(pt[:, 0]))
    
    deadline = 135.0  # Safety margin before 150s kill
    start_time = time.time()
    
    # Generate multiple candidate solutions
    candidates = []
    
    # 1. NEH heuristic
    t0 = time.time()
    neh_perm = _neh_order(pt)
    candidates.append(neh_perm)
    
    # 2. Palmer's key
    if time.time() - start_time < deadline:
        palmer_keys = _palmer_key(pt)
        palmer_perm = list(np.argsort(palmer_keys))
        candidates.append(palmer_perm)
    
    # 3. Longest total processing time (original)
    lpt_perm = list(np.argsort(-pt.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 4. Shortest total processing time
    spt_perm = list(np.argsort(pt.sum(axis=1)))
    candidates.append(spt_perm)
    
    # 5. Random permutations (a few)
    rng = np.random.default_rng(123)
    for _ in range(3):
        rand_perm = list(rng.permutation(n))
        candidates.append(rand_perm)
    
    # Evaluate all candidates and pick the best
    best_perm = candidates[0]
    best_ms = _makespan_fast(best_perm, pt)
    
    for perm in candidates[1:]:
        ms = _makespan_fast(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Local search on the best candidate
    if time.time() - start_time < deadline - 15.0:
        ls_budget = min(10.0, deadline - 15.0 - (time.time() - start_time))
        best_perm = _local_search(best_perm, pt, time_limit=ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    # Simulated annealing on the best candidate
    remaining = deadline - 5.0 - (time.time() - start_time)
    if remaining > 2.0:
        sa_budget = min(5.0, remaining)
        sa_perm = _simulated_annealing(best_perm, pt, time_limit=sa_budget)
        sa_ms = _makespan_fast(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Final local search if time permits
    remaining = deadline - 2.0 - (time.time() - start_time)
    if remaining > 1.0:
        final_ls_budget = min(3.0, remaining)
        best_perm = _local_search(best_perm, pt, time_limit=final_ls_budget)
        best_ms = _makespan_fast(best_perm, pt)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
