"""Flow-shop makespan minimizer: NEH + fast local search with multiple moves."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """
    Calculate makespan for a permutation using the standard recurrence.
    perm: 1-D array of job indices (length n_jobs)
    pt: (n_jobs, n_machines) processing times
    Returns: float makespan
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    # c[j] = completion time of job j on the current machine
    c = np.zeros(n_jobs, dtype=np.float64)
    p = pt[perm]  # (n_jobs, n_machines)
    
    for m in range(n_machines):
        c[0] = p[0, m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _makespan_delta_swap(perm, pt, i, j):
    """
    Compute the makespan change for swapping positions i and j (i < j)
    without recomputing the full makespan. This is complex, so we fall back
    to full computation for correctness. For performance, we use a fast
    incremental approach.
    """
    # For simplicity and correctness, we'll compute the full makespan.
    # A true delta computation would require tracking the critical path,
    # which is complex. We use the full computation but optimize the
    # inner loop.
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    p = pt[perm]
    c = np.zeros(n_jobs, dtype=np.float64)
    
    for m in range(n_machines):
        c[0] = p[0, m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _makespan_incremental(perm, pt):
    """
    Optimized makespan calculation using vectorized operations where possible.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    p = pt[perm]  # (n_jobs, n_machines)
    
    # We need to compute the flow-shop schedule.
    # c[j, m] = max(c[j, m-1], c[j-1, m]) + p[j, m]
    # We can compute this row by row (machine by machine).
    
    c = np.zeros(n_jobs, dtype=np.float64)
    for m in range(n_machines):
        if m == 0:
            c = np.cumsum(p[:, 0])
        else:
            # c[j] = max(c[j], c[j-1]) + p[j, m]
            # This is a sequential dependency, but we can use a trick:
            # c[j] = max(c[j], c[j-1]) + p[j, m]
            # Let d[j] = c[j] + p[j, m]
            # Then c[j] = max(c[j], c[j-1]) + p[j, m] = max(c[j] + p[j,m], c[j-1] + p[j,m])
            # This doesn't simplify easily. We'll keep the loop.
            for j in range(1, n_jobs):
                c[j] = max(c[j], c[j-1]) + p[j, m]
            c[0] += p[0, m]  # Wait, this is wrong. c[0] should be max(c[0], 0) + p[0,m] = c[0] + p[0,m]
            # Actually, for j=0: c[0] = max(c[0], 0) + p[0, m] = c[0] + p[0, m] (since c[0] >= 0)
            # But we need to be careful. Let me redo this.
            
    # Let me rewrite this more carefully.
    c = np.zeros(n_jobs, dtype=np.float64)
    for m in range(n_machines):
        c[0] += p[0, m]  # For j=0, c[0] = c[0] + p[0,m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _neh(processing_times):
    """
    NEH heuristic with fast insertion cost evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            # Build candidate permutation
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan_incremental(cand, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _local_search(perm, pt, deadline, margin):
    """
    Local search with swap, insert, and 2-opt moves.
    Uses fast makespan evaluation.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    best = perm.copy()
    best_cost = _makespan_incremental(best, pt)
    
    improved = True
    moves = []
    
    while improved:
        improved = False
        
        # Swap moves
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _makespan_incremental(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
                break
        
        if improved or time.monotonic() > deadline - margin:
            continue
        
        # 2-Opt (segment reversal) moves
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i:j+1] = cand[i:j+1][::-1]
                cost = _makespan_incremental(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
                break
        
        if improved or time.monotonic() > deadline - margin:
            continue
        
        # Insert moves: remove job at i and insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                break
            for j in range(n_jobs):
                if j == i:
                    continue
                cand = list(best)
                job = cand.pop(i)
                # Adjust j if i < j
                insert_pos = j if j <= i else j - 1
                cand.insert(insert_pos, job)
                cand = np.array(cand, dtype=np.int64)
                cost = _makespan_incremental(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
                break
    
    return best, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 145.0
    margin = 5.0

    # 1. NEH heuristic
    best_perm = _neh(processing_times)
    best_cost = _makespan_incremental(best_perm, processing_times)
    
    # 2. Local search
    best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 3. Try other initial solutions
    # SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    spt_cost = _makespan_incremental(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # SPT on last machine
    spt_last_perm = np.argsort(processing_times[:, -1])
    spt_last_cost = _makespan_incremental(spt_last_perm, processing_times)
    if spt_last_cost < best_cost:
        best_perm = spt_last_perm
        best_cost = spt_last_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.monotonic() > deadline - margin:
            break
        rand_perm = rng.permutation(n_jobs).astype(np.int64)
        rand_cost = _makespan_incremental(rand_perm, processing_times)
        if rand_cost < best_cost:
            best_perm = rand_perm
            best_cost = rand_cost
            best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
