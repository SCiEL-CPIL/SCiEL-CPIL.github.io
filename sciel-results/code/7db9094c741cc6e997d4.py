# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan for a permutation. O(n*m)."""
    n = len(perm)
    m = pt.shape[1]
    if m == 0 or n == 0:
        return 0.0
    
    prev_completion = np.cumsum(pt[perm, 0])
    for k in range(1, m):
        curr_completion = np.empty(n, dtype=np.float64)
        curr_completion[0] = prev_completion[0] + pt[perm[0], k]
        for i in range(1, n):
            curr_completion[i] = max(prev_completion[i], curr_completion[i-1]) + pt[perm[i], k]
        prev_completion = curr_completion
    
    return float(prev_completion[-1])


def _completion_matrix(perm, pt):
    """Compute full completion time matrix for all jobs and machines."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0 or m == 0:
        return np.zeros((n, 0))
    
    C = np.zeros((n, m), dtype=np.float64)
    C[:, 0] = np.cumsum(pt[perm, 0])
    for k in range(1, m):
        C[0, k] = C[0, k-1] + pt[perm[0], k]
        for i in range(1, n):
            C[i, k] = max(C[i, k-1], C[i-1, k]) + pt[perm[i], k]
    return C


def _makespan_from_C(C):
    """Get makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return float(C[-1, -1])


def _neh_heuristic(pt):
    """NEH heuristic for flow shop. Returns permutation."""
    n, m = pt.shape
    
    # Sort jobs by total processing time in descending order
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation greedily
    perm = []
    
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        
        if not perm:
            perm = [job]
            continue
        
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return np.array(perm, dtype=np.int64)


def _local_search(perm, pt, max_iters=200, deadline=None):
    """2-opt local search with incremental makespan update."""
    n = len(perm)
    if n <= 1:
        return perm
    
    best_perm = perm.copy()
    C = _completion_matrix(best_perm, pt)
    best_ms = _makespan_from_C(C)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() >= deadline:
            break
        
        for i in range(n - 1):
            if deadline is not None and time.time() >= deadline:
                break
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_perm = best_perm.copy()
                new_perm[i:j+1] = best_perm[i:j+1][::-1]
                
                # Recompute completion matrix for new permutation
                new_C = _completion_matrix(new_perm, pt)
                new_ms = _makespan_from_C(new_C)
                
                if new_ms < best_ms - 1e-9:
                    best_perm = new_perm
                    C = new_C
                    best_ms = new_ms
                    improved = True
    
    return best_perm


def _simulated_annealing(perm, pt, deadline, max_iters=5000):
    """Simulated annealing with incremental makespan evaluation."""
    n = len(perm)
    if n <= 1:
        return perm
    
    current_perm = perm.copy()
    current_C = _completion_matrix(current_perm, pt)
    current_ms = _makespan_from_C(current_C)
    
    best_perm = current_perm.copy()
    best_C = current_C.copy()
    best_ms = current_ms
    
    start_temp = max(1.0, current_ms * 0.01)
    end_temp = 1e-6
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    
    temp = start_temp
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        if temp < end_temp:
            break
        
        # Generate neighbor: random swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        new_perm = current_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        
        new_C = _completion_matrix(new_perm, pt)
        new_ms = _makespan_from_C(new_C)
        
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_perm = new_perm
            current_C = new_C
            current_ms = new_ms
            
            if current_ms < best_ms - 1e-9:
                best_perm = current_perm.copy()
                best_C = current_C.copy()
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Start with NEH heuristic
    best_perm = _neh_heuristic(pt)
    best_ms = _makespan(best_perm, pt)
    
    # Local search
    if time.time() < deadline:
        ls_perm = _local_search(best_perm, pt, max_iters=200, deadline=deadline)
        ls_ms = _makespan(ls_perm, pt)
        if ls_ms < best_ms:
            best_perm = ls_perm
            best_ms = ls_ms
    
    # Try multiple NEH variants with different sorting criteria
    if time.time() < deadline:
        # Sort by average processing time
        avgs = pt.mean(axis=1)
        order2 = np.argsort(-avgs)
        perm2_final = []
        for job in order2:
            best_pos = 0
            best_ms2 = float('inf')
            if not perm2_final:
                perm2_final = [job]
                continue
            for pos in range(len(perm2_final) + 1):
                new_perm = perm2_final[:pos] + [job] + perm2_final[pos:]
                ms = _makespan(np.array(new_perm, dtype=np.int64), pt)
                if ms < best_ms2:
                    best_ms2 = ms
                    best_pos = pos
            perm2_final.insert(best_pos, job)
        perm2_arr = np.array(perm2_final, dtype=np.int64)
        ms2 = _makespan(perm2_arr, pt)
        if ms2 < best_ms:
            best_perm = perm2_arr
            best_ms = ms2
        if time.time() < deadline:
            ls_perm2 = _local_search(best_perm, pt, max_iters=100, deadline=deadline)
            ls_ms2 = _makespan(ls_perm2, pt)
            if ls_ms2 < best_ms:
                best_perm = ls_perm2
                best_ms = ls_ms2
    
    # Simulated Annealing
    if time.time() < deadline - 5.0:
        sa_perm = _simulated_annealing(best_perm, pt, deadline, max_iters=3000)
        sa_ms = _makespan(sa_perm, pt)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
    
    # Random restarts if time permits
    rng = np.random.default_rng(42)
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() >= deadline - 2.0:
                break
            random_perm = rng.permutation(n)
            ms_r = _makespan(random_perm, pt)
            if ms_r < best_ms:
                best_perm = random_perm
                best_ms = ms_r
            # Local search from random start
            ls_r = _local_search(random_perm, pt, max_iters=50, deadline=deadline)
            ls_ms_r = _makespan(ls_r, pt)
            if ls_ms_r < best_ms:
                best_perm = ls_r
                best_ms = ls_ms_r
    
    return list(best_perm)
# EVOLVE-BLOCK-END
