"""Flow-shop permutation constructor with CDS initialization + multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, perm):
    """Compute makespan for a given permutation on a flow shop.
    pt: (n_jobs, n_machines) processing times
    perm: list/array of job indices in processing order
    Returns: makespan (float)
    """
    n_machines = pt.shape[1]
    # Cumulative completion times per machine
    # C[m] = completion time on machine m for the last job processed
    C = np.zeros(n_machines, dtype=np.float64)
    for j in perm:
        for m in range(n_machines):
            C[m] = max(C[m], C[m - 1] if m > 0 else 0.0) + pt[j, m]
    return C[-1]


def _makespan_fast(pt, perm):
    """Vectorized-ish makespan computation."""
    n_machines = pt.shape[1]
    # Extract submatrix for the permutation
    sub = pt[perm]  # (n, m)
    # For each machine m, completion time is sum over jobs of max(prev_machine_completion, current_machine_start)
    # Actually the standard recurrence:
    # C[j][m] = C[j][m-1] + pt[j][m]  (if j>0)
    # C[j][m] = C[j-1][m] + pt[j][m]  (if m>0)
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[j][m]  (if j>0 and m>0)
    # We can compute this iteratively but let's use a simple loop since n is likely small
    C = np.zeros((len(perm), n_machines), dtype=np.float64)
    for j in range(len(perm)):
        for m in range(n_machines):
            if j == 0 and m == 0:
                C[j, m] = sub[j, m]
            elif j == 0:
                C[j, m] = C[j, m - 1] + sub[j, m]
            elif m == 0:
                C[j, m] = C[j - 1, m] + sub[j, m]
            else:
                C[j, m] = max(C[j, m - 1], C[j - 1, m]) + sub[j, m]
    return C[-1, -1]


def _cds_init(pt):
    """CDS (Campbell, Dudek, Smith) heuristic for flow shop.
    Returns the best permutation among the three CDS sequences.
    """
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # SPT: sort by total processing time ascending
    spt = list(np.argsort(pt.sum(axis=1)))
    # LPT: sort by total processing time descending
    lpt = list(np.argsort(-pt.sum(axis=1)))
    # Reverse of SPT
    rev_spt = list(reversed(spt))
    
    candidates = [spt, lpt, rev_spt]
    
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan_fast(pt, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    return best_perm, best_ms


def _insertion_improve(pt, perm, max_passes=50, deadline=None):
    """First-improvement insertion local search.
    Try to improve by removing each job and inserting it at the best position.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(pt, perm)
    
    best_perm = list(perm)
    best_ms = _makespan_fast(pt, best_perm)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Remove job at position i
            job = best_perm[i]
            new_perm = best_perm[:i] + best_perm[i+1:]
            # Try all insertion positions
            best_pos = i
            best_ms_try = best_ms
            for pos in range(len(new_perm) + 1):
                if pos == i:
                    continue  # same position, skip
                trial = new_perm[:pos] + [job] + new_perm[pos:]
                ms_try = _makespan_fast(pt, trial)
                if ms_try < best_ms_try:
                    best_ms_try = ms_try
                    best_pos = pos
            if best_ms_try < best_ms:
                # Apply the best insertion
                new_perm = best_perm[:i] + best_perm[i+1:]
                best_perm = new_perm[:best_pos] + [job] + new_perm[best_pos:]
                best_ms = best_ms_try
                improved = True
        if not improved:
            break
    
    return best_perm, best_ms


def _interchange_improve(pt, perm, max_passes=50, deadline=None):
    """First-improvement interchange (swap) local search."""
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(pt, perm)
    
    best_perm = list(perm)
    best_ms = _makespan_fast(pt, best_perm)
    
    for _pass in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                # Swap i and j
                trial = list(best_perm)
                trial[i], trial[j] = trial[j], trial[i]
                ms_try = _makespan_fast(pt, trial)
                if ms_try < best_ms:
                    best_perm = trial
                    best_ms = ms_try
                    improved = True
                    break  # restart inner loop after a move
            if improved:
                break
        if not improved:
            break
    
    return best_perm, best_ms


def _perturb(pt, perm, strength=0.3, rng=None):
    """Apply a random perturbation: reverse a random segment and/or swap jobs."""
    if rng is None:
        rng = random.Random()
    n = len(perm)
    if n <= 2:
        return list(perm)
    
    perturbed = list(perm)
    
    # Random segment reversal
    seg_len = max(2, int(n * strength))
    start = rng.randint(0, n - seg_len)
    end = start + seg_len - 1
    perturbed[start:end+1] = perturbed[start:end+1][::-1]
    
    return perturbed


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow shop instance.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    Returns:
        list of job indices (permutation)
    """
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s kill (or 120s task limit)
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initialize with CDS
    best_perm, best_ms = _cds_init(pt)
    
    # Local search: insertion
    best_perm, best_ms = _insertion_improve(pt, best_perm, max_passes=100, deadline=deadline)
    
    # Check time
    if time.time() > deadline:
        return best_perm
    
    # Local search: interchange
    best_perm, best_ms = _interchange_improve(pt, best_perm, max_passes=100, deadline=deadline)
    
    # Check time
    if time.time() > deadline:
        return best_perm
    
    # Multi-start: perturb and re-optimize
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    
    if remaining_time > 5:
        # Determine how many restarts we can afford
        # Estimate cost per restart: insertion + interchange on current best
        # For small instances, each restart is fast
        max_restarts = min(20, int(remaining_time / 2))
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Perturb the best solution
            perturbed = _perturb(pt, best_perm, strength=0.3, rng=rng)
            
            # Local search on perturbed
            cand_perm, cand_ms = _insertion_improve(pt, perturbed, max_passes=50, deadline=deadline)
            cand_perm, cand_ms = _interchange_improve(pt, cand_perm, max_passes=50, deadline=deadline)
            
            if cand_ms < best_ms:
                best_perm = cand_perm
                best_ms = cand_ms
            elif time.time() > deadline:
                break
        
        # Final polish on the best
        if time.time() < deadline - 2:
            best_perm, best_ms = _insertion_improve(pt, best_perm, max_passes=100, deadline=deadline)
            if time.time() < deadline - 1:
                best_perm, best_ms = _interchange_improve(pt, best_perm, max_passes=100, deadline=deadline)
    
    return best_perm
# EVOLVE-BLOCK-END
