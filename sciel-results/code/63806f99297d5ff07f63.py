import time
import bisect
import array

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = 0
    
    primes = [i for i in range(limit + 1) if is_prime[i]]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # Precompute bisect_left results for all (i, j) pairs where i < j
    # bisect_L[i][j] = bisect_left(primes, primes[i] - (primes[j] - primes[i]), 0, i)
    # This is the smallest k < i such that primes[k] >= 2*primes[i] - primes[j]
    bisect_L = [[0] * N for _ in range(N)]
    for j in range(1, N):
        pj = primes[j]
        for i in range(0, j):
            pi = primes[i]
            threshold = pi - (pj - pi)  # 2*pi - pj
            bisect_L[i][j] = bisect.bisect_left(primes, threshold, 0, i)
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j] (i < j)
    # Use array('H') for memory efficiency (max length < 65535)
    dp = [array.array('H', [0] * N) for _ in range(N)]
    prev = [array.array('h', [-1] * N) for _ in range(N)]
    
    # suf_max_val[i] and suf_max_idx[i] for column i
    # suf_max_val[i][k] = max(dp[m][i]) for m in [k, i-1]
    # suf_max_idx[i][k] = argmax m in [k, i-1]
    suf_max_val = [None] * N
    suf_max_idx = [None] * N
    
    max_len = 1
    best_i = 0
    best_j = 0
    
    for j in range(1, N):
        # Check time every 100 iterations of j
        if j % 100 == 0 and time.time() > deadline:
            break
            
        # Compute dp[i][j] for all i < j
        dp_j = dp[j]
        prev_j = prev[j]
        
        for i in range(0, j):
            L = bisect_L[i][j]
            
            if L < i:
                # We have valid k in [L, i-1]
                # Get max from suf_max_val[i]
                val = suf_max_val[i][L]
                idx = suf_max_idx[i][L]
                
                dp_i_j = val + 1
                dp[i][j] = dp_i_j
                prev[i][j] = idx
            else:
                dp_i_j = 2
                dp[i][j] = 2
                prev[i][j] = -1
            
            if dp_i_j > max_len:
                max_len = dp_i_j
                best_i = i
                best_j = j
        
        # Compute suf_max for column j
        # Column j has entries dp[0][j], ..., dp[j-1][j]
        if j > 0:
            sm_val = [0] * j
            sm_idx = [0] * j
            
            # Initialize with last element m = j-1
            sm_val[j-1] = dp[j-1][j]
            sm_idx[j-1] = j-1
            
            for k in range(j-2, -1, -1):
                if dp[k][j] > sm_val[k+1]:
                    sm_val[k] = dp[k][j]
                    sm_idx[k] = k
                else:
                    sm_val[k] = sm_val[k+1]
                    sm_idx[k] = sm_idx[k+1]
            
            suf_max_val[j] = sm_val
            suf_max_idx[j] = sm_idx
    
    # Reconstruct sequence
    seq = [primes[best_j]]
    curr_i = best_i
    curr_j = best_j
    
    steps = 0
    while curr_i != -1 and steps < N:
        seq.append(primes[curr_i])
        prev_i = prev[curr_i][curr_j]
        curr_j = curr_i
        curr_i = prev_i
        steps += 1
    
    seq.reverse()
    
    return seq
