"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a permutation of job indices on flow-shop."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0:
        return 0.0
    if n_jobs == 0:
        return 0.0
    
    # Vectorized approach: accumulate completion times
    # prev[m] = completion time of previous job on machine m
    prev = np.zeros(n_machines)
    for job in perm:
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pt[job, 0]
        if n_machines > 1:
            # curr[m] = max(curr[m-1], prev[m]) + pt[job, m]
            # Can't fully vectorize due to sequential dependency within a job
            for m in range(1, n_machines):
                curr[m] = max(curr[m-1], prev[m]) + pt[job, m]
        prev = curr
    return prev[-1]


def _neh_heuristic(pt):
    """NEH (Nawaz, Enscore, Ham) heuristic for flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    # Sort jobs by total processing time descending
    totals = pt.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Insert each job into best position
    perm = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _makespan_fast(perm, pt):
    """Optimized makespan computation."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    prev = np.zeros(n_machines)
    for job in perm:
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pt[job, 0]
        if n_machines > 1:
            for m in range(1, n_machines):
                v1 = curr[m-1]
                v2 = prev[m]
                curr[m] = (v1 if v1 > v2 else v2) + pt[job, m]
        prev = curr
    return prev[-1]


def _local_search_insertion(perm, pt, deadline=None):
    """Insertion-based local search. Try moving each job to every other position."""
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    improved = True
    
    while improved:
        improved = False
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                break
            job = best_perm[i]
            # Try all insertion positions
            for j in range(n_jobs):
                if i == j:
                    continue
                # Remove job from i, insert at j
                candidate = best_perm[:i] + best_perm[i+1:]
                insert_pos = j if j < i else j - 1
                candidate = candidate[:insert_pos] + [job] + candidate[insert_pos:]
                ms = _makespan_fast(candidate, pt)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
                    break
            if deadline is not None and time.time() > deadline:
                break
        if deadline is not None and time.time() > deadline:
            break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan_fast(best_perm, pt)
    
    # Local search with insertion moves
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    # If we have time, try multiple restarts with different initial heuristics
    if time.time() < deadline - 10.0 and n_jobs > 5:
        # Try SPT (Shortest Processing Time) on first machine as alternative start
        spt_order = list(np.argsort(pt[:, 0]))
        if spt_order != best_perm:
            alt_perm = list(spt_order)
            alt_ms = _makespan_fast(alt_perm, pt)
            if time.time() < deadline - 5.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
        
        # Try LPT on last machine
        lpt_last = list(np.argsort(-pt[:, -1]))
        if lpt_last != best_perm:
            alt_perm = list(lpt_last)
            alt_ms = _makespan_fast(alt_perm, pt)
            if time.time() < deadline - 5.0:
                alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
        
        # Try Johnson's rule for 2-machine equivalent
        if n_machines >= 2 and time.time() < deadline - 10.0:
            # Use sum of first half and second half machines for Johnson's rule
            n_m = n_machines
            mid = n_m // 2
            a = pt[:, :mid].sum(axis=1)
            b = pt[:, mid:].sum(axis=1)
            johnson_order = list(np.argsort(np.where(a < b, a, -b), kind='stable'))
            # Actually Johnson's rule: sort by min(a_i, b_i), if a_i < b_i put early, else late
            johnson_order = []
            left = []
            right = []
            for j in range(n_jobs):
                if a[j] <= b[j]:
                    left.append((a[j], j))
                else:
                    right.append((b[j], j))
            left.sort(key=lambda x: x[0])
            right.sort(key=lambda x: -x[0])
            johnson_order = [x[1] for x in left] + [x[1] for x in right]
            
            if johnson_order != best_perm:
                alt_perm = list(johnson_order)
                alt_ms = _makespan_fast(alt_perm, pt)
                if time.time() < deadline - 5.0:
                    alt_perm, alt_ms = _local_search_insertion(alt_perm, pt, deadline)
                if alt_ms < best_ms - 1e-12:
                    best_perm = alt_perm
                    best_ms = alt_ms
    
    # Final polish: one more pass of local search if time remains
    if time.time() < deadline - 3.0:
        best_perm, best_ms = _local_search_insertion(best_perm, pt, deadline)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
