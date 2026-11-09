"""Flow-shop makespan minimizer: NEH + Fast Local Search with Delta Evaluation + Tabu + SA."""
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


def _delta_2opt(C, pt, i, j):
    """
    Compute delta makespan for reversing segment [i, j] (inclusive).
    Only the sub-matrix C[i:j+1, :] changes. We recompute that portion.
    Returns new makespan (not delta) for efficiency in comparison.
    """
    n_jobs = C.shape[0]
    n_machines = C.shape[1]
    
    # The jobs affected are positions i through j
    # We need to reverse the segment and recompute completions for those positions
    # and all subsequent positions
    
    # Create a copy of the relevant portion
    # Jobs before i are unchanged
    # Jobs from i to j are reversed
    # Jobs after j may change
    
    # Get the processing times for the segment
    # perm[i:j+1] reversed -> we need the actual job indices
    # But we only have C, not perm. So this approach won't work directly.
    # We need perm as well.
    
    return None  # Placeholder - we'll use a different approach


def _eval_2opt(perm, pt, C, i, j):
    """
    Evaluate the makespan after reversing segment [i, j] in perm.
    Uses the completion matrix C for fast recomputation.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    if i >= j:
        return _makespan_from_C(C)
    
    # Only need to recompute from position i onwards
    # Jobs 0..i-1 are unchanged
    
    # Create a 1D array to track the "incoming" completion times from previous position
    # For each machine m, we track the completion time of the job at position k-1 on machine m
    prev_row = C[i-1, :].copy() if i > 0 else np.zeros(n_machines, dtype=float)
    
    # Track the makespan as we go
    makespan = C[-1, -1]  # Will be updated
    
    # We'll compute new completion times row by row
    # new_C[k, m] for k >= i
    # We only need to track the current row and the previous row
    
    # For positions i to n_jobs-1
    # Job at position k:
    #   if k <= j: perm[i + j - k]
    #   else: perm[k]
    
    # Let's compute and track
    cur_row = np.zeros(n_machines, dtype=float)
    
    for k in range(i, n_jobs):
        if k <= j:
            job_idx = perm[i + j - k]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            if m == 0:
                prev_machine = 0.0
            else:
                prev_machine = cur_row[m-1]
            
            if k == 0:
                prev_job = 0.0
            else:
                prev_job = prev_row[m]
            
            cur_row[m] = max(prev_machine, prev_job) + p_job[m]
        
        # Update prev_row for next iteration
        if k < n_jobs - 1:
            prev_row = cur_row.copy()
    
    return cur_row[-1]


def _eval_swap(perm, pt, C, i, j):
    """
    Evaluate the makespan after swapping positions i and j in perm.
    Only positions i and j change, but all subsequent positions may be affected.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    if i == j:
        return _makespan_from_C(C)
    
    # Swap jobs at positions i and j
    # New job at position i: perm[j]
    # New job at position j: perm[i]
    # All other positions unchanged
    
    # Recompute from position min(i,j) onwards
    start = min(i, j)
    
    prev_row = C[start-1, :].copy() if start > 0 else np.zeros(n_machines, dtype=float)
    cur_row = np.zeros(n_machines, dtype=float)
    
    for k in range(start, n_jobs):
        if k == i:
            job_idx = perm[j]
        elif k == j:
            job_idx = perm[i]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            if m == 0:
                prev_machine = 0.0
            else:
                prev_machine = cur_row[m-1]
            
            if k == 0:
                prev_job = 0.0
            else:
                prev_job = prev_row[m]
            
            cur_row[m] = max(prev_machine, prev_job) + p_job[m]
        
        if k < n_jobs - 1:
            prev_row = cur_row.copy()
    
    return cur_row[-1]


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
    
    # Determine the starting position for recomputation
    if i < j:
        start_pos = i
    else:
        start_pos = j
    
    prev_row = C[start_pos-1, :].copy() if start_pos > 0 else np.zeros(n_machines, dtype=float)
    cur_row = np.zeros(n_machines, dtype=float)
    
    for k in range(start_pos, n_jobs):
        if i < j:
            if k < j:
                job_idx = perm[k + 1]  # shifted left
            elif k == j:
                job_idx = perm[i]      # the moved job
            else:
                job_idx = perm[k]      # unchanged
        else:
            if k == j:
                job_idx = perm[i]      # the moved job
            elif k > j and k <= i:
                job_idx = perm[k - 1]  # shifted right
            else:
                job_idx = perm[k]      # unchanged
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            if m == 0:
                prev_machine = 0.0
            else:
                prev_machine = cur_row[m-1]
            
            if k == 0:
                prev_job = 0.0
            else:
                prev_job = prev_row[m]
            
            cur_row[m] = max(prev_machine, prev_job) + p_job[m]
        
        if k < n_jobs - 1:
            prev_row = cur_row.copy()
    
    return cur_row[-1]


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
                    # Apply the move
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
                    # Apply insertion
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


def _tabu_search(perm, pt, deadline, margin, tabu_len=7):
    """
    Tabu search with insertion moves to escape local optima.
    """
    n_jobs, n_machines = pt.shape
    if n_jobs < 2:
        return perm, _calc_makespan(perm, pt)
    
    best_perm = perm.copy()
    best_cost = _calc_makespan(best_perm, pt)
    C = _completion_matrix(best_perm, pt)
    
    # Tabu list: store (i, j) pairs of recent moves
    tabu_list = []
    tabu_set = set()
    
    # Track the best solution found
    it = 0
    no_improve = 0
    max_no_improve = 50
    
    while time.monotonic() < deadline - margin:
        it += 1
        if no_improve >= max_no_improve:
            break
        
        best_move = None
        best_move_cost = float('inf')
        best_move_i = -1
        best_move_j = -1
        
        # Try all insertion moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                key = (i, j)
                if key in tabu_set:
                    continue
                
                new_cost = _eval_insert(best_perm, pt, C, i, j)
                
                # Aspiration: allow tabu move if it improves best
                if new_cost < best_cost - 1e-12 or (key not in tabu_set and new_cost < best_move_cost):
                    if new_cost < best_move_cost:
                        best_move_cost = new_cost
                        best_move_i = i
                        best_move_j = j
        
        if best_move_i == -1:
            # No valid moves, break
            break
        
        # Apply the best move
        i, j = best_move_i, best_move_j
        job = best_perm[i]
        if i < j:
            best_perm[i:j] = best_perm[i+1:j+1]
            best_perm[j] = job
        else:
            best_perm[j+1:i+1] = best_perm[j:i]
            best_perm[j] = job
        
        # Update tabu list
        tabu_list.append((i, j))
        tabu_set.add((i, j))
        if len(tabu_list) > tabu_len:
            old = tabu_list.pop(0)
            tabu_set.discard(old)
        
        # Recompute C
        C = _completion_matrix(best_perm, pt)
        
        # Update best
        if best_move_cost < best_cost - 1e-12:
            best_cost = best_move_cost
            no_improve = 0
        else:
            no_improve += 1
    
    return best_perm, best_cost


def _simulated_annealing(perm, pt, deadline, margin, rng):
    """
    Simulated annealing with insertion moves.
    """
    n_jobs, n_machines = pt.shape
    if n_jobs < 2:
        return perm, _calc_makespan(perm, pt)
    
    current_perm = perm.copy()
    current_cost = _calc_makespan(current_perm, pt)
    C = _completion_matrix(current_perm, pt)
    
    best_perm = current_perm.copy()
    best_cost = current_cost
    
    # Temperature schedule
    T0 = max(current_cost * 0.1, 1.0)
    Tend = max(current_cost * 0.001, 0.01)
    cooling_rate = 0.995
    
    T = T0
    step = 0
    
    while T > Tend and time.monotonic() < deadline - margin:
        step += 1
        
        # Generate a random insertion move
        i = rng.integers(0, n_jobs)
        j = rng.integers(0, n_jobs)
        if i == j:
            continue
        
        new_cost = _eval_insert(current_perm, pt, C, i, j)
        delta = new_cost - current_cost
        
        # Accept or reject
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Apply the move
            job = current_perm[i]
            if i < j:
                current_perm[i:j] = current_perm[i+1:j+1]
                current_perm[j] = job
            else:
                current_perm[j+1:i+1] = current_perm[j:i]
                current_perm[j] = job
            
            C = _completion_matrix(current_perm, pt)
            current_cost = new_cost
            
            if new_cost < best_cost - 1e-12:
                best_cost = new_cost
                best_perm = current_perm.copy()
        
        T *= cooling_rate
    
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
    rng = np.random.default_rng(42)

    # 1. Constructive Heuristic: NEH
    best_perm = _neh(processing_times)
    best_cost = _calc_makespan(best_perm, processing_times)
    
    # 2. Local Search with fast evaluation
    best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 3. If time remains, try alternative heuristics
    # SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    spt_cost = _calc_makespan(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 4. Tabu search to escape local optima
    if time.monotonic() < deadline - margin:
        tabu_perm, tabu_cost = _tabu_search(best_perm, processing_times, deadline, margin)
        if tabu_cost < best_cost - 1e-12:
            best_perm = tabu_perm
            best_cost = tabu_cost
    
    # 5. Simulated annealing for diversification
    if time.monotonic() < deadline - margin:
        sa_perm, sa_cost = _simulated_annealing(best_perm, processing_times, deadline, margin, rng)
        if sa_cost < best_cost - 1e-12:
            best_perm = sa_perm
            best_cost = sa_cost
    
    # 6. Random restarts with local search
    for _ in range(10):
        if time.monotonic() > deadline - margin:
            break
        rand_perm = rng.permutation(n_jobs)
        rand_cost = _calc_makespan(rand_perm, processing_times)
        if rand_cost < best_cost:
            best_perm = rand_perm
            best_cost = rand_cost
            best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 7. Final polish: run local search again on the best solution
    if time.monotonic() < deadline - margin:
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
