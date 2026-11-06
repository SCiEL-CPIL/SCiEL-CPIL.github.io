import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve of Eratosthenes
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    K = len(primes)
    
    if K < 2:
        return primes
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index h such that dp[h][i] was used to compute dp[i][j]
    # If no such h, prev[i][j] = -1
    # Base case: dp[i][j] = 2 for all i < j
    
    dp = np.full((K, K), 2, dtype=np.int32)
    prev = np.full((K, K), -1, dtype=np.int32)
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    for i in range(1, K - 1):
        if time.time() > deadline:
            break
            
        # Build suffix max and argmax for dp[h][i] for h in [0, i-1]
        col_i = dp[:i, i].copy()
        
        suffix_max = np.empty(i, dtype=np.int32)
        suffix_idx = np.empty(i, dtype=np.int32)
        suffix_max[i-1] = col_i[i-1]
        suffix_idx[i-1] = i-1
        
        for h in range(i-2, -1, -1):
            if col_i[h] > suffix_max[h+1]:
                suffix_max[h] = col_i[h]
                suffix_idx[h] = h
            else:
                suffix_max[h] = suffix_max[h+1]
                suffix_idx[h] = suffix_idx[h+1]
        
        # Vectorized computation for all j > i
        gaps = primes_arr[i+1:K] - primes_arr[i]
        targets = primes_arr[i] - gaps
        h_mins = np.searchsorted(primes_arr, targets, side='left')
        
        valid_mask = h_mins < i
        
        if np.any(valid_mask):
            valid_h_mins = h_mins[valid_mask]
            best_lens = suffix_max[valid_h_mins] + 1
            best_hs = suffix_idx[valid_h_mins]
            
            j_indices = np.arange(i+1, K)
            valid_j_indices = j_indices[valid_mask]
            
            dp[i, valid_j_indices] = best_lens
            prev[i, valid_j_indices] = best_hs
            
            max_local = int(np.max(best_lens))
            if max_local > best_len:
                best_len = max_local
                local_idx = int(np.argmax(best_lens))
                best_j = int(valid_j_indices[local_idx])
                best_i = i
    
    # Backtrack
    seq = [primes[best_j]]
    i = best_i
    j = best_j
    while i != -1:
        seq.append(primes[i])
        h = int(prev[i][j])
        if h == -1:
            break
        j = i
        i = h
    
    seq.reverse()
    
    return seq
