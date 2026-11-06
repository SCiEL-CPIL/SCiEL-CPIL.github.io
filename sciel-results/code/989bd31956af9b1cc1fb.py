import time
import math
import numpy as np

def construct():
    deadline = time.time() + 80
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    N = len(primes)
    
    if N <= 1:
        return primes
    
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    dp = np.full((N, N), 2, dtype=np.int32)
    parent = np.full((N, N), -1, dtype=np.int32)
    
    # suffix_max[i][k] = max(dp[k][i], ..., dp[i-1][i])
    # suffix_parent[i][k] = argmax index
    suffix_max = [None] * N
    suffix_parent = [None] * N
    
    # k_min[i] tracks the smallest valid k for current j
    k_min = list(range(N))
    
    best_len = 2
    best_end = (0, 1)
    
    primes_arr = np.array(primes, dtype=np.int64)
    
    # j is the index of the second prime in the pair
    for j in range(1, N):
        # Compute suffix_max for i = j-1
        i = j - 1
        if i > 0:
            col = dp[:i, i]
            # Use np.maximum.accumulate for suffix max
            sm_rev = np.maximum.accumulate(col[::-1])[::-1]
            suffix_max[i] = sm_rev
            # For parent: find argmax in suffix
            # sm_rev[k] = max(col[k], col[k+1], ..., col[i-1])
            # We need the index of the max in col[k:]
            sp = np.empty(i, dtype=np.int32)
            sp[i-1] = i-1
            for k in range(i-2, -1, -1):
                if col[k] > col[sp[k+1]]:
                    sp[k] = k
                else:
                    sp[k] = sp[k+1]
            suffix_parent[i] = sp
        else:
            suffix_max[i] = np.array([], dtype=np.int32)
            suffix_parent[i] = np.array([], dtype=np.int32)
            
        # Now compute dp[i][j] for all i < j (vectorized)
        pj = primes[j]
        # Vectorized: compute thresholds for all i < j
        thresholds = 2 * primes_arr[:j] - pj
        km_all = np.searchsorted(primes_arr, thresholds, side='left')
        # Clip to k_min[i]
        km_all = np.minimum(km_all, np.array(k_min[:j], dtype=np.int64))
        # Update k_min
        k_min[:j] = km_all.tolist()
        
        # For each i, if km_all[i] < i, we can use suffix_max[i][km_all[i]]
        valid_mask = km_all < np.arange(j)
        if np.any(valid_mask):
            idxs = np.where(valid_mask)[0]
            km_vals = km_all[idxs]
            # Get suffix_max values
            vals = np.empty(len(idxs), dtype=np.int32)
            for t, ii in enumerate(idxs):
                vals[t] = 1 + suffix_max[ii][km_vals[t]]
            # Update dp
            for t, ii in enumerate(idxs):
                if vals[t] > dp[ii, j]:
                    dp[ii, j] = vals[t]
                    parent[ii, j] = suffix_parent[ii][km_vals[t]]
                    if vals[t] > best_len:
                        best_len = vals[t]
                        best_end = (ii, j)
        
        # Check time periodically
        if j % 100 == 0 and time.time() > deadline:
            break
                
    # Reconstruct the sequence
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    while parent[ci, cj] != -1:
        ck = parent[ci, cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
        
    result.reverse()
    return result
