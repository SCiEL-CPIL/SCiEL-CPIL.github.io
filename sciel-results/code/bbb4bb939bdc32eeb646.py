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


def _completion_matrix(perm, pt):
    """Compute the full completion matrix for a given permutation."""
    n_machines = pt.shape[1]
    n_jobs = len(perm)
    
    if n_machines == 0 or n_jobs == 0:
        return np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    if n_jobs == 1:
        C = np.zeros((1, n_machines), dtype=np.float64)
        C[0, 0] = pt[perm[0], 0]
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt[perm[0], m]
        return C
    
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    pt_perm = pt[np.asarray(perm, dtype=np.int64)]
    
    C[:, 0] = np.cumsum(pt_perm[:, 0])
    
    if n_machines > 1:
        for m in range(1, n_machines):
            C[0, m] = C[0, m-1] + pt_perm[0, m]
            for j in range(1, n_jobs):
                C[j, m] = max(C[j, m-1], C[j-1, m]) + pt_perm[j, m]
    
    return C


def _makespan_from_matrix(C):
    """Extract makespan from a completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]


def _makespan_fast(perm, pt):
    """Optimized makespan computation using 2D array for speed."""
    C = _completion_matrix(perm, pt)
    return _makespan_from_matrix(C)


def _or_opt_insert(perm, C, pt, i, j, length):
    """
    Insert a block of jobs from position i (length jobs) into position j.
    Returns new completion matrix and makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    if i == j:
        return C, _makespan_from_matrix(C)
    
    # Extract the block to move
    block = list(perm[i:i+length])
    block_pt = pt[np.asarray(block, dtype=np.int64)]
    
    # Create new permutation
    new_perm = list(perm[:i] + perm[i+length:])
    # Adjust insertion position
    if j > i + length:
        new_j = j - length
    else:
        new_j = j
    
    new_perm = new_perm[:new_j] + block + new_perm[new_j:]
    
    # Recompute completion matrix for the new permutation
    new_C = _completion_matrix(new_perm, pt)
    return new_C, _makespan_from_matrix(new_C)


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


def _priority_heuristic(pt):
    """Priority-based heuristic using weighted machine priorities."""
    n_jobs = pt.shape[0]
    n_machines = pt.shape[1]
    
    if n_machines == 0:
        return list(range(n_jobs))
    
    # Weight machines by their position (later machines matter more for makespan)
    weights = np.linspace(1.0, 2.0, n_machines)
    priority = pt @ weights
    order = list(np.argsort(-priority))
    
    # Insert each job into best position using NEH-like insertion
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


def _local_search(perm, pt, deadline=None, use_swap=False, max_iterations=50):
    """Local search with Or-opt moves and swap moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_C = _completion_matrix(best_perm, pt)
    best_ms = _makespan_from_matrix(best_C)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Phase 1: Or-opt moves (move block of 1, 2, or 3 jobs)
        for block_len in [1, 2, 3]:
            if deadline is not None and time.time() > deadline:
                return best_perm, best_ms
            
            for i in range(n_jobs - block_len + 1):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                
                # Try inserting the block at all valid positions
                for j in range(n_jobs - block_len + 1):
                    if j == i:
                        continue
                    
                    new_C, new_ms = _or_opt_insert(best_perm, best_C, pt, i, j, block_len)
                    
                    if new_ms < best_ms - 1e-12:
                        best_ms = new_ms
                        best_C = new_C
                        # Reconstruct perm from the move
                        block = list(best_perm[i:i+block_len])
                        new_perm = list(best_perm[:i] + best_perm[i+block_len:])
                        if j > i + block_len:
                            new_j = j - block_len
                        else:
                            new_j = j
                        best_perm = new_perm[:new_j] + block + new_perm[new_j:]
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                break
        
        # Phase 2: swap moves if enabled and no improvement in Or-opt
        if use_swap and not improved and n_jobs > 1:
            for i in range(n_jobs):
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                for j in range(i + 1, n_jobs):
                    if deadline is not None and time.time() > deadline:
                        return best_perm, best_ms
                    
                    candidate = list(best_perm)
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    cand_C = _completion_matrix(candidate, pt)
                    ms = _makespan_from_matrix(cand_C)
                    
                    if ms < best_ms - 1e-12:
                        best_ms = ms
                        best_perm = candidate
                        best_C = cand_C
                        improved = True
                        break
                if deadline is not None and time.time() > deadline:
                    return best_perm, best_ms
                if improved:
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
    
    # 5. Priority-based heuristic
    priority_perm = _priority_heuristic(pt)
    existing_perms = [p for p, _ in initial_solutions]
    if priority_perm not in existing_perms:
        initial_solutions.append((priority_perm, _makespan_fast(priority_perm, pt)))
    
    # Sort initial solutions by makespan
    initial_solutions.sort(key=lambda x: x[1])
    
    best_perm, best_ms = initial_solutions[0]
    
    # Local search on best initial solution
    if time.time() < deadline - 5.0:
        best_perm, best_ms = _local_search(best_perm, pt, deadline, use_swap=True, max_iterations=100)
    
    # If we have time, try local search on other initial solutions
    # Sort remaining solutions by makespan and try them in order
    remaining = initial_solutions[1:]
    remaining.sort(key=lambda x: x[1])
    
    for alt_perm, alt_ms in remaining:
        if time.time() >= deadline - 10.0:
            break
        if alt_ms < best_ms + 0.15 * best_ms:  # Only if reasonably close
            alt_perm, alt_ms = _local_search(alt_perm, pt, deadline, use_swap=True, max_iterations=50)
            if alt_ms < best_ms - 1e-12:
                best_perm = alt_perm
                best_ms = alt_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
