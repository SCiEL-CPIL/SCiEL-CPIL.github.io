"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, perm):
    """Compute makespan for a given permutation using efficient loop."""
    n = len(perm)
    m = processing_times.shape[1]
    if n == 0 or m == 0:
        return 0.0
    # C[j][k] = max(C[j][k-1], C[j-1][k]) + p[perm[j]][k]
    # Use 1D array for current row, keep previous row
    prev_row = np.zeros(m)
    curr_row = np.zeros(m)
    p = processing_times  # alias for speed
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if j > 0 and k > 0:
                curr_row[k] = max(curr_row[k-1], prev_row[k])
            elif j > 0:
                curr_row[k] = prev_row[k]
            elif k > 0:
                curr_row[k] = curr_row[k-1]
            else:
                curr_row[k] = 0.0
            curr_row[k] += p[job, k]
        prev_row = curr_row.copy()
    return prev_row[-1]

def _neh_standard(processing_times):
    """Standard NEH: sort by total processing time descending."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    return _neh_insert(processing_times, sorted_jobs)

def _neh_first_last(processing_times):
    """NEH variant: interleave first and last machine processing times."""
    n, m = processing_times.shape
    if n == 0:
        return []
    if m == 1:
        totals = processing_times[:, 0]
        sorted_jobs = np.argsort(-totals)
        return _neh_insert(processing_times, sorted_jobs)
    # Key: max of first and last machine times
    keys = np.maximum(processing_times[:, 0], processing_times[:, -1])
    sorted_jobs = np.argsort(-keys)
    return _neh_insert(processing_times, sorted_jobs)

def _neh_last_machine(processing_times):
    """NEH variant: sort by last machine processing time descending."""
    n, m = processing_times.shape
    if n == 0:
        return []
    keys = processing_times[:, -1]
    sorted_jobs = np.argsort(-keys)
    return _neh_insert(processing_times, sorted_jobs)

def _neh_random(processing_times, rng):
    """Random NEH with noise on total processing times."""
    n, m = processing_times.shape
    if n == 0:
        return []
    totals = processing_times.sum(axis=1)
    noise = rng.uniform(0, 1.0, size=n)
    keys = -totals + noise * totals.max() * 0.1
    sorted_jobs = np.argsort(keys)
    return _neh_insert(processing_times, sorted_jobs)

def _neh_insert(processing_times, order):
    """Insert jobs in given order using best insertion position."""
    n = len(order)
    perm = []
    p = processing_times
    for job in order:
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(p, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    return perm

def _insertion_search(processing_times, perm, deadline=None):
    """First-improvement insertion local search."""
    n = len(perm)
    if n < 2:
        return list(perm)
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            job = current_perm[i]
            remaining = current_perm[:i] + current_perm[i+1:]
            for j in range(len(remaining) + 1):
                if j == i:
                    continue
                new_perm = remaining[:j] + [job] + remaining[j:]
                new_ms = _makespan(processing_times, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    return current_perm

def _or_opt_search(processing_times, perm, deadline=None):
    """First-improvement Or-opt local search (blocks of 1-3)."""
    n = len(perm)
    if n < 2:
        return list(perm)
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for block_len in [1, 2, 3]:
                if i + block_len > n:
                    continue
                block = current_perm[i:i+block_len]
                remaining = current_perm[:i] + current_perm[i+block_len:]
                for j in range(len(remaining) + 1):
                    if j == i or j == i + block_len:
                        continue
                    new_perm = remaining[:j] + block + remaining[j:]
                    new_ms = _makespan(processing_times, new_perm)
                    if new_ms < current_ms:
                        current_perm = new_perm
                        current_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return current_perm

def _swap_search(processing_times, perm, deadline=None):
    """First-improvement swap local search."""
    n = len(perm)
    if n < 2:
        return list(perm)
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                if deadline is not None and time.time() > deadline:
                    break
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                new_ms = _makespan(processing_times, new_perm)
                if new_ms < current_ms:
                    current_perm = new_perm
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    return current_perm

def _simulated_annealing(processing_times, perm, start_temp, end_temp, max_iter, deadline=None):
    """Simulated annealing for flow shop."""
    n = len(perm)
    if n < 2:
        return list(perm)
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Cooling schedule
        progress = it / max(max_iter - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Random move: swap two random positions
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        new_perm = current_perm[:]
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan(processing_times, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            current_perm = new_perm
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
    
    return best_perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (jobs, machines)
        
    Returns:
        List of job indices in the order they should be processed.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 seconds safety margin
    
    n, m = processing_times.shape
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions using multiple NEH variants
    candidates = []
    
    # Standard NEH
    perm1 = _neh_standard(processing_times)
    candidates.append(perm1)
    
    # First-last NEH
    perm2 = _neh_first_last(processing_times)
    candidates.append(perm2)
    
    # Last-machine NEH
    perm3 = _neh_last_machine(processing_times)
    candidates.append(perm3)
    
    # Random NEH restarts
    for _ in range(8):
        if time.time() > deadline:
            break
        perm_r = _neh_random(processing_times, rng)
        candidates.append(perm_r)
    
    # Evaluate all candidates and pick best
    best_perm = None
    best_ms = np.inf
    for perm in candidates:
        if time.time() > deadline:
            break
        ms = _makespan(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Local search: insertion
    if time.time() < deadline:
        improved = _insertion_search(processing_times, best_perm, deadline=deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    # Local search: Or-opt
    if time.time() < deadline:
        improved = _or_opt_search(processing_times, best_perm, deadline=deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    # Local search: swap
    if time.time() < deadline:
        improved = _swap_search(processing_times, best_perm, deadline=deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    # Simulated annealing to escape local optima
    if time.time() < deadline - 5:
        sa_start = time.time()
        sa_deadline = min(deadline, sa_start + 30)
        improved = _simulated_annealing(processing_times, best_perm, 100.0, 0.01, 5000, deadline=sa_deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    # Final polish: insertion search
    if time.time() < deadline - 2:
        improved = _insertion_search(processing_times, best_perm, deadline=deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    # Final polish: Or-opt
    if time.time() < deadline - 1:
        improved = _or_opt_search(processing_times, best_perm, deadline=deadline)
        ms = _makespan(processing_times, improved)
        if ms < best_ms:
            best_perm = improved
            best_ms = ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
