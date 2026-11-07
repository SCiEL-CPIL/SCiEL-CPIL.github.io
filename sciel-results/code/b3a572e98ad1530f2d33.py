import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using a sieve
    # sigma[n] = sum of divisors of n
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Initialize sigma with 1 for all n >= 1 (divisor 1)
    # sigma[0] remains 0
    sigma[1:] = 1
    
    # Sieve of Eratosthenes variant for sigma
    # For each prime p, add p^k * (1 + p + ... + p^k) to multiples of p^k
    # However, the simple divisor loop is often faster in NumPy for N=10^7 if optimized
    # Let's try the divisor loop with NumPy slicing as it is vectorized
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline:
        return [11, 6]

    # 2. Sort indices by sigma(n)
    n_values = np.arange(1, N + 1, dtype=np.int64)
    s_values = sigma[1:N+1]
    
    # Sort by sigma, then by n
    sort_idx = np.lexsort((n_values, s_values))
    sorted_n = n_values[sort_idx]
    sorted_s = s_values[sort_idx]
    
    if time.time() > deadline:
        return [11, 6]

    # 3. Find best pair
    best_score = float('inf')
    best_a, best_b = 11, 6
    
    i = 0
    N_total = len(sorted_n)
    
    while i < N_total:
        current_sigma = sorted_s[i]
        # Find the end of the group
        next_idx = np.searchsorted(sorted_s, current_sigma, side='right')
        
        if next_idx - i >= 2:
            # Extract the group of numbers with the same sigma
            group = sorted_n[i:next_idx]
            n_l = len(group)
            
            # Convert to list for bisect
            group_list = group.tolist()
            
            # For each b in group, find a in group minimizing |a/b - 1.5|
            # We iterate through b. To avoid O(k^2), we use bisect.
            # Since group is sorted, bisect is O(log k).
            for k in range(n_l):
                b = group_list[k]
                target_a = 1.5 * b
                
                # Find position in group
                pos = bisect_left(group_list, int(target_a))
                
                # Check neighbors
                if pos < n_l:
                    a = group_list[pos]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            current_score = diff / (2.0 * b)
                            if current_score < best_score:
                                best_score = current_score
                                best_a = a
                                best_b = b
                
                if pos > 0:
                    a = group_list[pos - 1]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            current_score = diff / (2.0 * b)
                            if current_score < best_score:
                                best_score = current_score
                                best_a = a
                                best_b = b
        
        i = next_idx
        
        if time.time() > deadline:
            break

    return [int(best_a), int(best_b)]
