import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 135s limit, 15s buffer before 150s kill
    
    # We want to find M with many k such that k*sigma(k)=M.
    # Strategy: sieve sigma(k) for k=1..K_max, compute M=k*sigma(k),
    # find M with most preimages.
    
    # Try progressively larger K_max. Start with a reasonable one,
    # if time permits, try larger.
    
    # Memory: sigma array is (K_max+1) * 8 bytes for int64.
    # K_max=5*10^7 -> 400MB. Should be OK.
    # K_max=1*10^8 -> 800MB. Risky but let's try.
    
    # Let's try K_max = 50_000_000 (50 million)
    # Sieve is O(N log N) ~ 50M * 17 = 850M ops.
    # NumPy slice ops are fast but the Python loop is the bottleneck.
    
    K_max = 50_000_000
    
    sigma = np.zeros(K_max + 1, dtype=np.int64)
    
    # Sieve: for each d, add d to sigma[d], sigma[2d], sigma[3d], ...
    # Check time every 1000 iterations to reduce overhead
    t_check_interval = 1000
    for d in range(1, K_max + 1):
        if d % t_check_interval == 0 and time.time() > deadline - 10.0:
            break
        sigma[d::d] += d
        
    elapsed_sieve = time.time() - start_time
    
    # Calculate M = k * sigma(k)
    k_arr = np.arange(1, K_max + 1, dtype=np.int64)
    M_arr = k_arr * sigma[1:]
    
    # Filter M <= 10^18
    valid_mask = M_arr <= 10**18
    
    M_valid = M_arr[valid_mask]
    k_valid = k_arr[valid_mask]
    
    # Group by M using np.unique
    unique_M, inverse = np.unique(M_valid, return_inverse=True)
    counts = np.bincount(inverse)
    
    # Find the M with max count
    max_idx = np.argmax(counts)
    max_M = unique_M[max_idx]
    max_count = counts[max_idx]
    
    # Retrieve k's for max_M
    k_solutions = k_valid[inverse == max_idx]
    k_list = k_solutions.tolist()
    
    if not k_list:
        return {'M': 1, 'k': [1]}
    
    return {'M': int(max_M), 'k': k_list}
