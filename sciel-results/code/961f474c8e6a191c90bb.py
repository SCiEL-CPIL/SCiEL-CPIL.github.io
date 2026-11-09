"""Flow-shop makespan minimizer: NEH + CIG constructive, 2-opt & swap local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan of a job permutation using cumulative sums.
    
    For a flow shop, the makespan is the completion time on the last machine.
    C[j][m] = max(C[j-1][m], C[j][m-1]) + p[j][m]
    The makespan is C[n-1][m-1].
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # Use a running array for efficiency
    # C[m] holds completion times for the current job on each machine
    C = np.zeros(m)
    for j in range(n):
        job = perm[j]
        for k in range(m):
            if k == 0:
                C[k] += pt[job, k]
            else:
                C[k] = max(C[k], C[k-1]) + pt[job, k]
    return float(C[-1])


def _makespan_fast(perm, pt):
    """Fast makespan using vectorized operations where possible."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    # This is the standard O(n*m) computation
    C = np.zeros(m)
    for j in range(n):
        job = perm[j]
        p_row = pt[job]
        for k in range(m):
            if k == 0:
                C[k] += p_row[k]
            else:
                C[k] = max(C[k], C[k-1]) + p_row[k]
    return float(C[-1])


def _neh(processing_times, seed=42):
    """NEH (Nawaz, Enscore, Ham) heuristic.
    
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence.
    """
    n, m = processing_times.shape
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    
    # Step 1: Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals, kind='stable')
    
    # Step 2: Insert each job into the best position
    seq = np.empty(n, dtype=np.int64)
    seq[0] = order[0]
    best_ms = _makespan(seq[:1], processing_times)
    
    for i in range(1, n):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        # Try inserting at each position 0..i
        for pos in range(i + 1):
            # Create new sequence
            new_seq = np.empty(i + 1, dtype=np.int64)
            new_seq[:pos] = seq[:pos]
            new_seq[pos] = job
            new_seq[pos+1:] = seq[pos:i]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Update sequence
        new_seq = np.empty(i + 1, dtype=np.int64)
        new_seq[:best_pos] = seq[:best_pos]
        new_seq[best_pos] = job
        new_seq[best_pos+1:] = seq[best_pos:i]
        seq[:i+1] = new_seq
    
    return seq


def _cig(processing_times, seed=42):
    """CIG (Campbell, Iggleston, and Jones) heuristic.
    
    1. Sort jobs by total processing time in descending order.
    2. For each job, insert it at the position that minimizes the makespan,
       but use a greedy forward insertion.
    """
    n, m = processing_times.shape
    if n == 0:
        return np.array([], dtype=np.int64)
    if n == 1:
        return np.array([0], dtype=np.int64)
    
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals, kind='stable')
    
    seq = np.empty(n, dtype=np.int64)
    seq[0] = order[0]
    
    for i in range(1, n):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(i + 1):
            new_seq = np.empty(i + 1, dtype=np.int64)
            new_seq[:pos] = seq[:pos]
            new_seq[pos] = job
            new_seq[pos+1:] = seq[pos:i]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        new_seq = np.empty(i + 1, dtype=np.int64)
        new_seq[:best_pos] = seq[:best_pos]
        new_seq[best_pos] = job
        new_seq[best_pos+1:] = seq[best_pos:i]
        seq[:i+1] = new_seq
    
    return seq


def _local_search_2opt(perm, pt, max_iter=1000, deadline=None):
    """2-opt local search: reverse a segment of the permutation."""
    n = len(perm)
    if n < 2:
        return perm
    
    perm = perm.copy()
    best_ms = _makespan(perm, pt)
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment [i..j]
                new_perm = perm.copy()
                new_perm[i:j+1] = perm[i:j+1][::-1]
                ms = _makespan(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    perm = new_perm
                    improved = True
                    if deadline is not None and time.time() > deadline:
                        return perm
    return perm


def _local_search_swap(perm, pt, max_iter=1000, deadline=None):
    """Swap local search: swap any two adjacent or non-adjacent jobs."""
    n = len(perm)
    if n < 2:
        return perm
    
    perm = perm.copy()
    best_ms = _makespan(perm, pt)
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            for j in range(i + 1, n):
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan(new_perm, pt)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    perm = new_perm
                    improved = True
                    if deadline is not None and time.time() > deadline:
                        return perm
    return perm


def _local_search_insert(perm, pt, max_iter=1000, deadline=None):
    """Insert local search: move a job to a different position."""
    n = len(perm)
    if n < 2:
        return perm
    
    perm = perm.copy()
    best_ms = _makespan(perm, pt)
    improved = True
    iters = 0
    
    while improved and iters < max_iter:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i in range(n):
            job = perm[i]
            for j in range(n):
                if i == j:
                    continue
                # Remove job from i and insert at j
                new_perm = np.empty(n, dtype=np.int64)
                # Copy without job at i
                idx = 0
                for k in range(n):
                    if k == i:
                        continue
                    new_perm[idx] = perm[k]
                    idx += 1
                # Insert job at position j (adjust if j > i)
                insert_pos = j if j < i else j
                # Actually, we need to be careful with the index
                # Let's just build it properly
                new_perm = np.empty(n, dtype=np.int64)
                # First, get the sequence without job
                seq_without = np.array([perm[k] for k in range(n) if k != i])
                # Now insert at position j (in the new sequence of length n-1)
                # If j >= i, the position in the new sequence is j-1 (since we removed one before)
                # Actually, let's just try all positions 0..n-1 in the new sequence
                for pos in range(n):
                    candidate = np.empty(n, dtype=np.int64)
                    if pos == 0:
                        candidate[0] = job
                        candidate[1:] = seq_without
                    elif pos == n - 1:
                        candidate[:-1] = seq_without
                        candidate[-1] = job
                    else:
                        candidate[:pos] = seq_without[:pos]
                        candidate[pos] = job
                        candidate[pos+1:] = seq_without[pos:]
                    ms = _makespan(candidate, pt)
                    if ms < best_ms - 1e-9:
                        best_ms = ms
                        perm = candidate
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if deadline is not None and time.time() > deadline:
            break
    return perm


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Generate initial solutions using different heuristics
    candidates = []
    
    # NEH
    neh_perm = _neh(pt)
    candidates.append(neh_perm)
    
    # CIG
    cig_perm = _cig(pt)
    candidates.append(cig_perm)
    
    # SPT (Shortest Processing Time first)
    spt_perm = np.argsort(pt.sum(axis=1), kind='stable')
    candidates.append(spt_perm)
    
    # LPT (Longest Processing Time first) - same as NEH initial sort
    # Random permutations
    rng = np.random.default_rng(42)
    for _ in range(5):
        random_perm = rng.permutation(n)
        candidates.append(random_perm)
    
    # Evaluate all candidates
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, pt)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Apply local search to the best candidate
    if time.time() < deadline - 5.0:
        # Try 2-opt
        perm_2opt = _local_search_2opt(best_perm, pt, max_iter=500, deadline=deadline - 2.0)
        ms_2opt = _makespan(perm_2opt, pt)
        if ms_2opt < best_ms:
            best_ms = ms_2opt
            best_perm = perm_2opt
    
    if time.time() < deadline - 5.0:
        # Try swap
        perm_swap = _local_search_swap(best_perm, pt, max_iter=500, deadline=deadline - 2.0)
        ms_swap = _makespan(perm_swap, pt)
        if ms_swap < best_ms:
            best_ms = ms_swap
            best_perm = perm_swap
    
    if time.time() < deadline - 5.0:
        # Try insert
        perm_insert = _local_search_insert(best_perm, pt, max_iter=500, deadline=deadline - 2.0)
        ms_insert = _makespan(perm_insert, pt)
        if ms_insert < best_ms:
            best_ms = ms_insert
            best_perm = perm_insert
    
    # Final check: ensure we're within deadline
    if time.time() > deadline:
        # Return the best we have
        pass
    
    return list(best_perm.astype(np.int64))
# EVOLVE-BLOCK-END
