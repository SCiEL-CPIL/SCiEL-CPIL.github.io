import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    hard_deadline = start_time + 110  # Leave 10s for final processing

    # 1. Run Sieve with NumPy
    # Use a reasonably large N; NumPy makes this fast
    N = 50_000_000
    
    sigma = np.zeros(N + 1, dtype=np.uint64)
    t_start = time.time()
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        # Check time every 5000 iterations
        if i % 5000 == 0:
            if time.time() > hard_deadline - 30:
                break
    
    elapsed_sieve = time.time() - t_start
    
    # 2. Find Best M using vectorized operations
    k_arr = np.arange(1, N + 1, dtype=np.uint64)
    M_arr = k_arr * sigma[1:N+1]
    
    # Filter valid M values
    valid_mask = M_arr <= np.uint64(10**18)
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    # Find the M with most solutions
    if len(M_valid) == 0:
        return {'M': 0, 'k': []}
    
    # Use sorting to group by M
    sort_idx = np.argsort(M_valid)
    M_sorted = M_valid[sort_idx]
    k_sorted = k_valid[sort_idx]
    
    # Find unique M values and their counts
    change_points = np.where(np.diff(M_sorted) != 0)[0]
    if len(change_points) == 0:
        best_M = int(M_sorted[0])
        best_k = k_sorted.tolist()
    else:
        # Find which group is largest
        group_starts = np.concatenate(([0], change_points + 1))
        group_ends = np.concatenate((change_points + 1, [len(M_sorted)]))
        group_sizes = group_ends - group_starts
        max_idx = np.argmax(group_sizes)
        
        best_M = int(M_sorted[group_starts[max_idx]])
        best_k = k_sorted[group_starts[max_idx]:group_ends[max_idx]].tolist()
    
    # Ensure types are correct (Python int, not numpy types)
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
