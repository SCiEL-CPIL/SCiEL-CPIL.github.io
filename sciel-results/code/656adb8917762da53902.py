import time
import math
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    
    N_LIMIT = 10000
    
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
            
    primes = np.array([i for i, val in enumerate(is_prime) if val])
    n = len(primes)
    
    if n < 2:
        return primes.tolist()
        
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    max_len = 0
    best_i = -1
    best_j = -1
    
    # j = 0
    dp[1:, 0] = 2
    max_len = 2
    best_i = 1
    best_j = 0
    
    for j in range(1, n):
        if time.time() - t0 > DEADLINE:
            break
            
        # Compute suffix max and argmax for dp[j, 0..j-1]
        suf_val = np.zeros(j, dtype=np.int32)
        suf_idx = np.zeros(j, dtype=np.int32)
        suf_val[j-1] = dp[j, j-1]
        suf_idx[j-1] = j-1
        for k in range(j-2, -1, -1):
            if dp[j, k] > suf_val[k+1]:
                suf_val[k] = dp[j, k]
                suf_idx[k] = k
            else:
                suf_val[k] = suf_val[k+1]
                suf_idx[k] = suf_idx[k+1]
        
        # Vectorized update for i = j+1..n-1
        targets = 2 * primes[j] - primes[j+1:]
        k_mins = np.searchsorted(primes[:j], targets, side='left')
        
        valid = k_mins < j
        dp_vals = np.where(valid, 1 + suf_val[k_mins[valid]], 2)
        prev_vals = np.where(valid, suf_idx[k_mins[valid]], -1)
        
        dp[j+1:, j] = dp_vals
        prev[j+1:, j] = prev_vals
        
        col_max = np.max(dp[j+1:, j])
        if col_max > max_len:
            idx = np.argmax(dp[j+1:, j])
            max_len = col_max
            best_i = j + 1 + idx
            best_j = j
    
    # Reconstruct
    if best_i == -1:
        return [int(primes[0])]
        
    seq = [int(primes[best_i]), int(primes[best_j])]
    cur_i = best_i
    cur_j = best_j
    while prev[cur_i, cur_j] != -1:
        cur_k = int(prev[cur_i, cur_j])
        seq.append(int(primes[cur_k]))
        cur_i, cur_j = cur_j, cur_k
        
    seq.reverse()
    return seq
