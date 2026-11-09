"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a permutation of job indices on flow-shop."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Efficient computation using 1D arrays
    prev = np.zeros(n_machines)
    for job in perm:
        curr = np.empty(n_machines)
        curr[0] = prev[0] + pt[job, 0]
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], prev[m]) + pt[job, m]
        prev = curr
    return prev[-1]


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Build completion matrix C[j][m] = completion time of job j on machine m
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    # First machine: cumulative sum
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                c_prev_m = C[j-1, m]
                c_curr_m1 = C[j, m-1]
                C[j, m] = (c_curr_m1 if c_curr_m1 > c_prev_m else c_prev_m) + pt_perm[j, m]
    
    return C[n_jobs-1, n_machines-1]


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
            ms = _makespan_fast(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _johnsons_rule(pt):
    """Johnson's rule generalized for multi-machine flow shop."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_machines <= 2:
        if n_machines == 2:
            left = []
            right = []
            for j in range(n_jobs):
                if pt[j, 0] <= pt[j, 1]:
                    left.append((pt[j, 0], j))
                else:
                    right.append((pt[j, 1], j))
            left.sort(key=lambda x: x[0])
            right.sort(key=lambda x: -x[0])
            return [x[1] for x in left] + [x[1] for x in right]
        else:
            return list(np.argsort(pt[:, 0]))
    
    # For multi-machine: use sum of first half and second half
    mid = n_machines // 2
    a = pt[:, :mid].sum(axis=1)
    b = pt[:, mid:].sum(axis=1)
    
    left = []
    right = []
    for j in range(n_jobs):
        if a[j] <= b[j]:
            left.append((a[j], j))
        else:
            right.append((b[j], j))
    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: -x[0])
    return [x[1] for x in left] + [x[1] for x in right]


def _local_search(perm, pt, deadline=None, use_swap=False):
    """Local search with insertion and optional swap moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_ms = _makespan_fast(best_perm, pt)
    improved = True
    
    while improved:
        improved = False
        
        # Phase 1: insertion moves
        for i in range(n_jobs):
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            job = best_perm[i]
            for j in range(n_jobs):
                if i == j:
                    continue
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
                return best_perm, best_ms
        
        # Phase 2: swap moves if enabled
        if use_swap and not improved and n_jobs > 1:
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                for j in range(i + 1, n_jobs):
                    if deadline is not None and time.time() > deadline:
                        return best_perm, best_ms
                    candidate = list(best_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    ms = _makespan_fast(candidate, pt)
                    if ms < best_ms - 1e-12:
                        best_ms = ms
                        best_perm = candidate
                        improved = True
                        break
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
    
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
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # 1. NEH heuristic
    neh_perm = _neh_heuristic(pt)
    initial_solutions.append((neh_perm, _makespan_fast(neh_perm, pt)))
    
    # 2. Johnson's rule
    johnson_perm = _johnsons_rule(pt)
    if johnson_perm != neh_perm:
        initial_solutions.append((johnson_perm, _makespan_fast(johnson_perm, pt)))
    
    # 3. SPT on first machine
    spt_order = list(np.argsort(pt[:, 0]))
    if spt_order != neh_perm and spt_order != johnson_perm:
        initial_solutions.append((spt_order, _makespan_fast(spt_order, pt)))
    
    # 4. LPT on last machine
    lpt_last = list(np.argsort(-pt[:, -1]))
    if lpt_last != neh_perm and lpt_last != johnson_perm and lpt_last != spt_order:
        initial_solutions.append((lpt_last, _makespan_fast(lpt_last, pt)))
    
    # Sort initial solutions by makespan
    initial_solutions.sort(key=lambda x: x[1])
    
    best_perm, best_ms = initial_solutions[0]
    
    # Local search on best initial solution
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search(best_perm, pt, deadline, use_swap=True)
    
    # If we have time, try local search on second-best initial solution
    if len(initial_solutions) > 1 and time.time() < deadline - 10.0:
        alt_perm, alt_ms = initial_solutions[1]
        if alt_ms < best_ms + 0.1 * best_ms:  # Only if reasonably close
            alt_perm, alt_ms = _local_search(alt_perm, pt, deadline, use_swap=True)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
    
    # If we have time, try local search on third-best initial solution
    if len(initial_solutions) > 2 and time.time() < deadline - 10.0:
        alt_perm, alt_ms = initial_solutions[2]
        if alt_ms < best_ms + 0.1 * best_ms:
            alt_perm, alt_ms = _local_search(alt_perm, pt, deadline, use_swap=True)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
