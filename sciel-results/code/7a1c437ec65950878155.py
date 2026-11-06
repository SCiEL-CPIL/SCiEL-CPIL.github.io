"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(seq, pt):
    """Compute makespan of a permutation on a flow shop.
    seq: sequence of job indices (list or array)
    pt:  np.ndarray [n_jobs, n_machines]
    """
    if len(seq) == 0:
        return 0.0
    n_m = pt.shape[1]
    # C[j][k] = max(C[j-1][k], C[j][k-1]) + p[seq[j]][k]
    # Keep only the previous row to save memory.
    prev = np.zeros(n_m, dtype=np.float64)
    for job in seq:
        row = pt[job]
        # current completion on machine k
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return float(prev[-1])

def _delta_move(seq, pt, i, j):
    """Fast evaluation of moving job at position i to position j.
    Only the affected segment needs recomputation.
    Returns the new makespan.
    """
    n = len(seq)
    if i == j:
        return _makespan(seq, pt)
    n_m = pt.shape[1]
    # Build the new sequence but only recompute the segment between min(i,j) and max(i,j)+1
    # Actually the dependency flows forward, so we need completion times up to the segment.
    # We compute the full completion array once up to the affected region, then adjust.
    # For simplicity and correctness, recompute from position min(i,j) onward,
    # but we need the completion state at position min(i,j)-1.
    
    start = min(i, j)
    # Compute completion times up to start-1
    if start == 0:
        prev = np.zeros(n_m, dtype=np.float64)
    else:
        prev = np.zeros(n_m, dtype=np.float64)
        for idx in range(start):
            row = pt[seq[idx]]
            curr = np.empty(n_m, dtype=np.float64)
            curr[0] = prev[0] + row[0]
            for k in range(1, n_m):
                a = prev[k]
                b = curr[k-1]
                curr[k] = (a if a > b else b) + row[k]
            prev = curr
    
    # Build the new sequence from start onward
    moved_job = seq[i]
    new_seq = list(seq)
    new_seq.pop(i)
    target = j - 1 if j > i else j
    new_seq.insert(target, moved_job)
    
    # Recompute from start
    for idx in range(start, n):
        row = pt[new_seq[idx]]
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return float(prev[-1])

def _neh_heuristic(pt):
    """NEH heuristic: sort by decreasing total processing time, insert each job
    at the best position."""
    n_jobs, n_machines = pt.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    total = pt.sum(axis=1)
    order = list(np.argsort(-total))
    
    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
    return seq

def _local_search(seq, pt, deadline, max_passes=50):
    """Local search with 2-opt (move) neighborhood, first-improvement strategy.
    Returns the best sequence found within the time budget."""
    n = len(seq)
    best_seq = list(seq)
    best_ms = _makespan(best_seq, pt)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            job = best_seq[i]
            for j in range(n):
                if i == j:
                    continue
                new_ms = _delta_move(best_seq, pt, i, j)
                if new_ms < best_ms - 1e-9:
                    # Apply the move
                    best_seq.pop(i)
                    target = j - 1 if j > i else j
                    best_seq.insert(target, job)
                    best_ms = new_ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_seq, best_ms

def _swap_neighbor(seq, pt, i, j):
    """Swap jobs at positions i and j, return new makespan."""
    n = len(seq)
    n_m = pt.shape[1]
    start = min(i, j)
    if start == 0:
        prev = np.zeros(n_m, dtype=np.float64)
    else:
        prev = np.zeros(n_m, dtype=np.float64)
        for idx in range(start):
            row = pt[seq[idx]]
            curr = np.empty(n_m, dtype=np.float64)
            curr[0] = prev[0] + row[0]
            for k in range(1, n_m):
                a = prev[k]
                b = curr[k-1]
                curr[k] = (a if a > b else b) + row[k]
            prev = curr
    new_seq = list(seq)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    for idx in range(start, n):
        row = pt[new_seq[idx]]
        curr = np.empty(n_m, dtype=np.float64)
        curr[0] = prev[0] + row[0]
        for k in range(1, n_m):
            a = prev[k]
            b = curr[k-1]
            curr[k] = (a if a > b else b) + row[k]
        prev = curr
    return float(prev[-1])

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. NEH construction
    seq = _neh_heuristic(processing_times)
    best_ms = _makespan(seq, processing_times)
    
    # 2. Local search with move neighborhood
    seq, best_ms = _local_search(seq, processing_times, deadline)
    
    # 3. Try swap neighborhood as a second pass
    if time.time() < deadline:
        n = len(seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    new_ms = _swap_neighbor(seq, processing_times, i, j)
                    if new_ms < best_ms - 1e-9:
                        seq[i], seq[j] = seq[j], seq[i]
                        best_ms = new_ms
                        improved = True
                        break
                if improved:
                    break
        # Re-run move local search after swaps
        if improved and time.time() < deadline:
            seq, best_ms = _local_search(seq, processing_times, deadline)
    
    return seq

# EVOLVE-BLOCK-END
