"""Flow-shop makespan minimizer: NEH + Fast Local Search with Delta Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _calc_makespan(perm, pt):
    """
    Calculate the makespan for a given permutation.
    perm: array of job indices
    pt: processing times (n_jobs, n_machines)
    Returns: makespan (float)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    p = pt[perm]
    c = np.zeros(n_jobs, dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                start_time = c[j]
            else:
                start_time = max(c[j], c[j-1])
            c[j] = start_time + p[j, m]
            
    return c[-1]


def _completion_matrix(perm, pt):
    """
    Compute the full completion time matrix C[j, m] = completion time of job at position j on machine m.
    Returns C of shape (n_jobs, n_machines).
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return np.empty((0, n_machines))
    
    p = pt[perm]
    C = np.zeros((n_jobs, n_machines), dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                C[j, m] = (C[j, m-1] if m > 0 else 0.0) + p[j, m]
            else:
                C[j, m] = max(C[j, m-1] if m > 0 else 0.0, C[j-1, m]) + p[j, m]
    
    return C


def _makespan_from_C(C):
    """Extract makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]


def _eval_2opt(perm, pt, C, i, j):
    """
    Evaluate the makespan after reversing segment [i, j] in perm.
    Uses the completion matrix C for fast recomputation.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    new_C = C.copy()
    
    for k in range(i, n_jobs):
        if k <= j:
            job_idx = perm[i + j - k]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _eval_swap(perm, pt, C, i, j):
    """
    Evaluate the makespan after swapping positions i and j in perm.
    Only positions i and j change, but all subsequent positions may be affected.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    new_C = C.copy()
    
    for k in range(i, n_jobs):
        if k == i:
            job_idx = perm[j]
        elif k == j:
            job_idx = perm[i]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _eval_insert(perm, pt, C, i, j):
    """
    Evaluate the makespan after removing job at position i and inserting at position j.
    If i < j: job moves from i to j, jobs i+1..j shift left by 1
    If i > j: job moves from i to j, jobs j..i-1 shift right by 1
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    if i == j:
        return _makespan_from_C(C)
    
    new_C = C.copy()
    
    if i < j:
        start_pos = i
    else:
        start_pos = j
    
    for k in range(start_pos, n_jobs):
        if i < j:
            if k < j:
                job_idx = perm[k + 1]
            elif k == j:
                job_idx = perm[i]
            else:
                job_idx = perm[k]
        else:
            if k == j:
                job_idx = perm[i]
            elif k > j and k <= i:
                job_idx = perm[k - 1]
            else:
                job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _neh(processing_times):
    """
    NEH heuristic: order jobs by total processing time desc, insert each optimally.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = np.empty(n_jobs, dtype=np.int64)
    perm[0] = order[0]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            cand = np.empty(i + 1, dtype=np.int64)
            if pos == 0:
                cand[0] = job
                cand[1:] = perm[:i]
            elif pos == i:
                cand[:i] = perm[:i]
                cand[i] = job
            else:
                cand[:pos] = perm[:pos]
                cand[pos] = job
                cand[pos + 1:] = perm[pos:i]
            
            cost = _calc_makespan(cand, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        new_perm = np.empty(i + 1, dtype=np.int64)
        if best_pos == 0:
            new_perm[0] = job
            new_perm[1:] = perm[:i]
        elif best_pos == i:
            new_perm[:i] = perm[:i]
            new_perm[i] = job
        else:
            new_perm[:best_pos] = perm[:best_pos]
            new_perm[best_pos] = job
            new_perm[best_pos + 1:] = perm[best_pos:i]
        
        perm[:i + 1] = new_perm
        
    return perm


def _local_search(perm, processing_times, deadline, margin):
    """
    Local search using 2-opt, swap, and insertion moves with fast delta evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    C = _completion_matrix(best, processing_times)
    best_cost = _makespan_from_C(C)
    improved = True
    
    while improved:
        improved = False
        
        # 2-Opt: Reverse a segment [i, j]
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_2opt(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Swap: Exchange two positions i, j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_swap(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    best[i], best[j] = best[j], best[i]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Insertion: Remove job at i, insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if i == j:
                    continue
                new_cost = _eval_insert(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    job = best[i]
                    if i < j:
                        best[i:j] = best[i+1:j+1]
                        best[j] = job
                    else:
                        best[j+1:i+1] = best[j:i]
                        best[j] = job
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
                
    return best, best_cost


def _simulated_annealing(perm, processing_times, deadline, margin):
    """
    Simulated Annealing to escape local optima.
    Uses delta evaluation for speed.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs < 3:
        return perm.copy(), _calc_makespan(perm, processing_times)
    
    current = perm.copy()
    C = _completion_matrix(current, processing_times)
    current_cost = _makespan_from_C(C)
    
    best = current.copy()
    best_C = C.copy()
    best_cost = current_cost
    
    t0 = time.monotonic()
    initial_temp = max(1.0, current_cost * 0.01)
    cooling_rate = 0.995
    
    temp = initial_temp
    
    while temp > 1e-6:
        if time.monotonic() > deadline - margin:
            break
        
        # Pick a random move: insertion
        i = np.random.randint(n_jobs)
        j = np.random.randint(n_jobs)
        if i == j:
            continue
        
        new_cost = _eval_insert(current, processing_times, C, i, j)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            # Accept move
            job = current[i]
            if i < j:
                current[i:j] = current[i+1:j+1]
                current[j] = job
            else:
                current[j+1:i+1] = current[j:i]
                current[j] = job
            
            C = _completion_matrix(current, processing_times)
            current_cost = new_cost
            
            if current_cost < best_cost - 1e-12:
                best = current.copy()
                best_C = C.copy()
                best_cost = current_cost
        
        temp *= cooling_rate
    
    return best, best_cost


def _multistart(processing_times, deadline, margin):
    """
    Multi-start local search with diverse initial solutions.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    neh_perm = _neh(processing_times)
    candidates.append(neh_perm)
    
    # 2. SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    candidates.append(spt_perm)
    
    # 3. LPT on last machine
    lpt_perm = np.argsort(-processing_times[:, -1])
    candidates.append(lpt_perm)
    
    # 4. Random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        rand_perm = rng.permutation(n_jobs)
        candidates.append(rand_perm)
    
    best_perm = None
    best_cost = float('inf')
    
    for cand in candidates:
        if time.monotonic() > deadline - margin:
            break
        
        # Local search from this candidate
        ls_perm, ls_cost = _local_search(cand, processing_times, deadline, margin)
        
        if ls_cost < best_cost:
            best_perm = ls_perm
            best_cost = ls_cost
    
    # Simulated annealing from the best
    if time.monotonic() < deadline - margin:
        sa_perm, sa_cost = _simulated_annealing(best_perm, processing_times, deadline, margin)
        if sa_cost < best_cost:
            best_perm = sa_perm
            best_cost = sa_cost
    
    return best_perm, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 145.0  # Safety margin
    margin = 5.0

    # Multi-start with diverse heuristics + local search + simulated annealing
    best_perm, best_cost = _multistart(processing_times, deadline, margin)
    
    # Final local search polish
    if time.monotonic() < deadline - margin:
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
