"""Iterated Local Search for flow-shop permutation problem."""
# EVOLVE-BLOCK-START
import numpy as np

def _makespan_fast(order, p):
    """Compute makespan for a given order using vectorized-ish approach."""
    n_m = p.shape[1]
    if len(order) == 0:
        return 0.0
    completion = np.zeros(n_m)
    for job in order:
        row = p[job]
        c = completion[0] + row[0]
        completion[0] = c
        for m in range(1, n_m):
            c = max(c, completion[m]) + row[m]
            completion[m] = c
    return completion[-1]

def _delta_makespan(seq, p, i, j):
    """
    Compute new makespan after moving job from position i to position j (i < j).
    Uses prefix/suffix completion arrays for O(n_m) per affected position.
    """
    n = len(seq)
    n_m = p.shape[1]
    if i == j:
        return _makespan_fast(seq, p)
    
    # Build prefix completions: after processing jobs 0..k, completion times
    # prefix[k][m] = completion time on machine m after job k
    prefix = np.zeros((n, n_m))
    for k in range(n):
        row = p[seq[k]]
        c = prefix[k-1][0] + row[0] if k > 0 else row[0]
        prefix[k][0] = c
        for m in range(1, n_m):
            c = max(c, prefix[k-1][m]) + row[m] if k > 0 else max(0, 0) + row[m]
            prefix[k][m] = c
    
    # Suffix: for each position k, the completion if we started from position k
    # with initial completion = prefix[k-1]
    # Actually, let's just recompute the middle part.
    
    # Move job from i to j
    moved_job = seq[i]
    middle_jobs = seq[i+1:j+1]  # these shift right by 1
    
    # Before: positions 0..i-1, then i (moved), then i+1..j, then j+1..
    # After: 0..i-1, then i+1..j (shifted), then moved at j, then j+1..
    
    # Completion after position i-1
    if i > 0:
        start_comp = prefix[i-1].copy()
    else:
        start_comp = np.zeros(n_m)
    
    # Process middle_jobs (shifted right)
    comp = start_comp.copy()
    for job in middle_jobs:
        row = p[job]
        c = comp[0] + row[0]
        comp[0] = c
        for m in range(1, n_m):
            c = max(c, comp[m]) + row[m]
            comp[m] = c
    
    # Now process moved_job at position j
    row = p[moved_job]
    c = comp[0] + row[0]
    comp[0] = c
    for m in range(1, n_m):
        c = max(c, comp[m]) + row[m]
        comp[m] = c
    
    # Process remaining jobs j+1..n-1
    for k in range(j+1, n):
        row = p[seq[k]]
        c = comp[0] + row[0]
        comp[0] = c
        for m in range(1, n_m):
            c = max(c, comp[m]) + row[m]
            comp[m] = c
    
    return comp[-1]

def _local_search_2opt(seq, p, max_passes=50):
    """
    Local search using 2-opt (move one job to another position).
    Returns improved sequence and makespan.
    """
    n = len(seq)
    current = seq[:]
    current_ms = _makespan_fast(current, p)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                new_ms = _delta_makespan(current, p, i, j)
                if new_ms < current_ms:
                    # Move job from i to j
                    job = current[i]
                    current = current[:i] + current[i+1:j+1] + [job] + current[j+1:]
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current, current_ms

def _neh_initial(p):
    """NEH heuristic: sort by total processing time descending, insert optimally."""
    n, m = p.shape
    totals = p.sum(axis=1)
    order = np.argsort(-totals)
    
    seq = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_fast(cand, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, int(job))
    return seq

def _neh_inverse_initial(p):
    """Inverse NEH: sort by total processing time ascending."""
    n, m = p.shape
    totals = p.sum(axis=1)
    order = np.argsort(totals)
    
    seq = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            cand = seq[:pos] + [job] + seq[pos:]
            ms = _makespan_fast(cand, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, int(job))
    return seq

def _random_initial(n):
    return list(np.random.permutation(n))

def _perturb(seq, n_pert):
    """Perturb a sequence by n_pert random moves."""
    n = len(seq)
    s = seq[:]
    for _ in range(n_pert):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        job = s[i]
        s = s[:i] + s[i+1:]
        if j > i:
            j -= 1
        s = s[:j] + [job] + s[j:]
    return s

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    best_global_seq = []
    best_global_ms = float('inf')
    
    # Generate multiple initial solutions
    initial_solutions = []
    
    # NEH
    initial_solutions.append(_neh_initial(processing_times))
    
    # Inverse NEH
    initial_solutions.append(_neh_inverse_initial(processing_times))
    
    # Random starts
    n_random = 3 if n_jobs <= 15 else 2
    for _ in range(n_random):
        initial_solutions.append(_random_initial(n_jobs))
    
    # Run local search on each initial solution
    for init_seq in initial_solutions:
        ls_seq, ls_ms = _local_search_2opt(init_seq, processing_times, max_passes=30)
        if ls_ms < best_global_ms:
            best_global_ms = ls_ms
            best_global_seq = ls_seq
    
    # Iterated local search: perturb best and re-optimize
    n_iters = 10 if n_jobs <= 15 else 5
    for it in range(n_iters):
        # Perturb the best solution
        n_pert = max(2, n_jobs // 4)
        perturbed = _perturb(best_global_seq, n_pert)
        
        ls_seq, ls_ms = _local_search_2opt(perturbed, processing_times, max_passes=30)
        if ls_ms < best_global_ms:
            best_global_ms = ls_ms
            best_global_seq = ls_seq
    
    return best_global_seq
# EVOLVE-BLOCK-END
