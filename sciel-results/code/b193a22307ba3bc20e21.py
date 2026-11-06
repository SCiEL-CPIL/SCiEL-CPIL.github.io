<<<<<<< SEARCH
import numpy as np
import time


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Compute cumulative completion times
    # For each job in order, compute completion on each machine
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    # where i is job index in permutation, j is machine index
    
    # Vectorized approach:
    # completion[i, j] = max(completion[i, j-1], completion[i-1, j]) + P[perm[i], j]
    # completion[-1, j] = P[perm[0], j] for first job
    
    P = processing_times[perm]  # (n_jobs, n_machines)
    
    if n_machines == 1:
        return float(P.sum())
    
    # Use iterative computation
    # completion[i, j] = max(completion[i, j-1], completion[i-1, j]) + P[i, j]
    # We can compute this row by row
    
    # completion for first job
    comp = np.zeros((n_jobs, n_machines))
    comp[0, 0] = P[0, 0]
    for j in range(1, n_machines):
        comp[0, j] = comp[0, j-1] + P[0, j]
    
    for i in range(1, n_jobs):
        comp[i, 0] = comp[i-1, 0] + P[i, 0]
        for j in range(1, n_machines):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + P[i, j]
    
    return float(comp[-1, -1])
=======
import numpy as np
import time


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation. Vectorized where possible."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    P = processing_times[perm]  # (n_jobs, n_machines)
    
    if n_machines == 1:
        return float(P.sum())
    
    if n_jobs == 1:
        return float(np.sum(P[0]))
    
    # For small problems, use the iterative approach which is clean
    # For larger, we can try to vectorize the machine dimension
    comp = np.empty((n_jobs, n_machines))
    comp[0, 0] = P[0, 0]
    if n_machines > 1:
        comp[0, 1:] = np.cumsum(P[0, 1:])
    
    # Process remaining jobs; the recurrence is:
    # comp[i, 0] = comp[i-1, 0] + P[i, 0]
    # comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + P[i, j]
    # We can compute each row's interior with a scan, but the dependency on
    # comp[i-1, j] makes full vectorization tricky. Use row-by-row with
    # vectorized interior.
    for i in range(1, n_jobs):
        comp[i, 0] = comp[i-1, 0] + P[i, 0]
        if n_machines > 1:
            # For j=1..n_machines-1: comp[i,j] = max(comp[i-1,j], comp[i,j-1]) + P[i,j]
            # This is a sequential scan along j, but we can do it in a tight loop
            # which is faster than Python-level max calls
            prev_row = comp[i-1, 1:]
            cur_row = comp[i, 1:]
            p_row = P[i, 1:]
            # Manual scan
            for j in range(1, n_machines):
                cur_row[j-1] = max(prev_row[j-1], cur_row[j-2]) + p_row[j-1]
    
    return float(comp[-1, -1])
>>>>>>> REPLACE

