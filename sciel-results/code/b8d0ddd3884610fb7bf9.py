import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, 20s buffer
    
    # We want to find M with many k such that k*sigma(k)=M.
    # We iterate k up to K_max.
    # K_max should be large enough to find collisions.
    # M = k*sigma(k) ~ k^2 * ln(ln(k)).
    # If M <= 10^18, k can be up to ~ 10^9.
    # But we are limited by time and memory.
    
    # Strategy: Use a two-phase approach
    # Phase 1: Quick sieve with smaller K_max to find candidate M values
    # Phase 2: If time remains, extend to larger K_max
    
    # Start with a moderate K_max
    K_max = 50_000_000  # 50 million - should be feasible with numpy
    
    # Compute sigma(k) for k = 1..K_max using the divisor sieve
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    
    # Time check: if we're running out of time, reduce K_max
    elapsed = time.time() - start_time
    if elapsed > 10:
        K_max = 30_000_000
        sigma = np.zeros(K_max + 1, dtype=np.int64)
    
    # Perform the sieve
    for d in range(1, K_max + 1):
        if time.time() > deadline - 10.0:
            break
        sigma[d::d] += d
        
    # Calculate M = k * sigma(k)
    k_arr = np.arange(1, K_max + 1, dtype=np.int64)
    M_arr = k_arr * sigma[1:]
    
    # Filter M <= 10^18
    valid_mask = M_arr <= 10**18
    
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    # Find the M with the most k values
    unique_M, inverse = np.unique(M_valid, return_inverse=True)
    counts = np.bincount(inverse)
    
    # Find the top few M values by count
    # Sort by count descending
    sorted_indices = np.argsort(counts)[::-1]
    
    # Get the best M and its k values
    max_idx = sorted_indices[0]
    max_M = unique_M[max_idx]
    max_count = counts[max_idx]
    
    k_solutions = k_valid[inverse == max_idx]
    k_list = k_solutions.tolist()
    
    if not k_list:
        return {'M': 1, 'k': [1]}
        
    return {'M': int(max_M), 'k': k_list}
