import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for safety
    
    # We want to find M with many k such that k * sigma(k) = M.
    # Strategy: Iterate k, compute M = k * sigma(k), store in dict M -> list of k.
    # k must be <= sqrt(M). Since M = k * sigma(k) >= k * (1+k) = k^2 + k, 
    # sqrt(M) >= sqrt(k^2+k) > k. So any k we find satisfies k < sqrt(M).
    # Actually, the constraint is k <= isqrt(M).
    # M = k * sigma(k). 
    # isqrt(k * sigma(k)) >= k  <=>  k * sigma(k) >= k^2  <=>  sigma(k) >= k.
    # sigma(k) >= 1 + k > k. So the condition k <= isqrt(M) is always satisfied for k >= 1.
    
    # We need to maximize the number of distinct k in the list for a single M.
    # We can iterate k up to some limit K_max.
    # M <= 10^18.
    # k * sigma(k) <= 10^18.
    
    # Use NumPy for faster sigma sieve computation
    # Try N = 50,000,000 with NumPy acceleration
    
    N = 50_000_000  # 5*10^7
    
    # Check time before starting heavy work
    if time.time() > deadline - 30.0:
        # Fallback to small search or known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Initialize sigma array using NumPy
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve using NumPy vectorized operations
    sig = sigma
    for i in range(2, N + 1):
        # Vectorized addition: sigma[i::i] += i
        sig[i::i] += i
        
        # Check time periodically
        if i % 1_000_000 == 0:
            if time.time() > deadline - 15.0:
                break
    
    if time.time() > deadline - 10.0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Now, bucket M values
    # M = k * sigma[k]
    # We only care if M <= 10^18
    # And we want to count distinct k for each M.
    
    buckets = defaultdict(list)
    limit_M = 10**18
    
    # Convert to Python list for faster iteration in Python loop
    # Or use NumPy vectorization for the bucketing
    
    # Iterate k and bucket
    # Use vectorized operations where possible
    
    # First, compute M = k * sigma[k] for all k
    k_array = np.arange(1, N + 1, dtype=np.int64)
    M_array = k_array * sig[1:N+1]
    
    # Filter: M <= 10^18
    valid_mask = M_array <= limit_M
    valid_k = k_array[valid_mask]
    valid_M = M_array[valid_mask]
    
    # Bucket using Python dict (NumPy doesn't have efficient groupby)
    # Convert to Python lists for the bucketing phase
    valid_k_list = valid_k.tolist()
    valid_M_list = valid_M.tolist()
    
    for k, M in zip(valid_k_list, valid_M_list):
        buckets[M].append(k)
        
        if time.time() > deadline - 5.0:
            # Time out, process current buckets
            break
            
    # Find best M
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Iterate through buckets
    for M, ks in buckets.items():
        cnt = len(ks)
        if cnt > best_count:
            best_count = cnt
            best_M = M
            best_ks = ks
            
    # Ensure types are correct
    # M is int, k is list of ints.
    
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
