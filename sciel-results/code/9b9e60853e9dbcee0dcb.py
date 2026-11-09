import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120  # Safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    
    # Candidate N values - larger N means more k values to check
    N_candidates = [5*10**6, 10*10**6, 15*10**6, 20*10**6, 30*10**6]
    
    for N in N_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
            
        # Compute sigma using vectorized approach
        # sigma[n] = sum of divisors of n
        # Use the fact that sigma is multiplicative and use sieve-like approach
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            # Add i to all multiples of i
            sigma[i::i] += i
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            # Not enough time to process, but we have sigma
            # Try to quickly find best
            pass
        
        # Find M with most k
        # Compute M = k * sigma[k] for all k
        k_array = np.arange(1, N + 1, dtype=np.int64)
        M_array = k_array * sigma[1:]
        
        # Filter M <= 10^18
        valid_mask = M_array <= 10**18
        M_valid = M_array[valid_mask]
        k_valid = k_array[valid_mask]
        
        # Group by M value - use sorting approach for efficiency
        # Sort by M value
        order = np.argsort(M_valid, kind='stable')
        M_sorted = M_valid[order]
        k_sorted = k_valid[order]
        
        # Find runs of equal M values
        # Use np.diff to find where M changes
        if len(M_sorted) > 0:
            # Find boundaries where M changes
            changes = np.where(np.diff(M_sorted) != 0)[0]
            boundaries = np.concatenate([[0], changes + 1, [len(M_sorted)]])
            
            max_count = 0
            current_best_M = None
            current_best_k_list = None
            
            for i in range(len(boundaries) - 1):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                count = end_idx - start_idx
                if count > max_count:
                    max_count = count
                    current_best_M = int(M_sorted[start_idx])
                    current_best_k_list = [int(k) for k in k_sorted[start_idx:end_idx]]
            
            if current_best_M is not None:
                if len(current_best_k_list) > len(best_k):
                    best_M = current_best_M
                    best_k = current_best_k_list
    
    return {'M': best_M, 'k': best_k}
