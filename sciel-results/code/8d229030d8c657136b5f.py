"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(processing_times, order):
    """
    Compute the makespan for a given job order.
    processing_times: np.ndarray of shape (n_jobs, n_machines)
    order: list of job indices
    Returns: float makespan
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    # C[i][j] is completion time of job at position i on machine j
    # We can compute this efficiently.
    # Let's use a vectorized approach if possible, or a tight loop.
    # For small N, a loop is fine. For large N, vectorization helps.
    
    # Extract processing times for the order
    # p[i][j] is processing time of job order[i] on machine j
    p = processing_times[order]
    
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
        
    # C[0, j] = sum_{k=0}^j p[0, k]
    # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
    
    # We can compute this row by row.
    # C is (n_jobs, n_machines)
    C = np.zeros((n_jobs, n_machines))
    
    # First row
    C[0, 0] = p[0, 0]
    if n_machines > 1:
        C[0, 1:] = np.cumsum(p[0, 1:])
    
    # Subsequent rows
    for i in range(1, n_jobs):
        C[i, 0] = C[i-1, 0] + p[i, 0]
        if n_machines > 1:
            # C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
            # This dependency makes it hard to fully vectorize the inner loop without tricks.
            # However, for typical N (e.g., < 100), this loop is fast enough.
            for j in range(1, n_machines):
                C[i, j] = max(C[i-1, j], C[i, j-1]) + p[i, j]
                
    return C[-1, -1]

def _neh_construction(processing_times):
    """
    Construct an initial sequence using the NEH heuristic.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    seq = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = np.inf
        
        # Try inserting job at each position in current seq
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        seq.insert(best_pos, job)
        
    return seq

def _local_search(processing_times, seq, time_limit, start_time):
    """
    Perform 2-swap local search.
    """
    n_jobs = len(seq)
    current_ms = _makespan(processing_times, seq)
    improved = True
    
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        # Try all pairs of swaps? That's O(N^2) evaluations per pass.
        # Each evaluation is O(N*M). Total O(N^3*M) per pass.
        # If N is large, this is too slow.
        # Let's restrict to adjacent swaps first, or limit passes.
        # Actually, 2-swap (any two) is stronger.
        # We can iterate i from 0 to N-1, j from i+1 to N-1.
        
        for i in range(n_jobs - 1):
            if (time.time() - start_time) > time_limit - 5:
                return seq, current_ms
            
            for j in range(i + 1, n_jobs):
                if i == 0 and j == n_jobs - 1:
                    # Skip swap if it doesn't change order (reversal? No, swap is just exchange)
                    # Swap i and j
                    pass
                
                # Swap i and j
                seq[i], seq[j] = seq[j], seq[i]
                new_ms = _makespan(processing_times, seq)
                
                if new_ms < current_ms:
                    current_ms = new_ms
                    improved = True
                else:
                    # Swap back
                    seq[i], seq[j] = seq[j], seq[i]
                    
    return seq, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    time_limit = 145  # Safety margin for 150s limit
    
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    
    # 1. NEH Construction
    seq = _neh_construction(processing_times)
    
    # 2. Local Search
    seq, _ = _local_search(processing_times, seq, time_limit, start_time)
    
    return seq
# EVOLVE-BLOCK-END
