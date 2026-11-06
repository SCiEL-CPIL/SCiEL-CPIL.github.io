import time
import math
import numpy as np

def construct():
    deadline = time.time() + 85
    
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
    
    primes_arr = np.array(primes, dtype=np.int32)
    
    # j is the index of the second prime in the pair
    for j in range(1, N):
        # Compute suffix_max for i = j-1
        i = j - 1
        if i > 0:
            sm = np.zeros(i, dtype=np.int32)
            sp = np.full(i, -1, dtype=np.int32)
            sm[i-1] = dp[i-1, i]
            sp[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[k, i] > sm[k+1]:
                    sm[k] = dp[k, i]
                    sp[k] = k
                else:
                    sm[k] = sm[k+1]
                    sp[k] = sp[k+1]
            suffix_max[i] = sm
            suffix_parent[i] = sp
        else:
            suffix_max[i] = np.array([], dtype=np.int32)
            suffix_parent[i] = np.array([], dtype=np.int32)
            
        # Now compute dp[i][j] for all i < j
        pj = primes[j]
        for i in range(j):
            # Update k_min[i]
            threshold = 2 * primes[i] - pj
            km = k_min[i]
            while km > 0 and primes[km - 1] >= threshold:
                km -= 1
            k_min[i] = km
            
            if km < i:
                val = 1 + suffix_max[i][km]
                if val > dp[i, j]:
                    dp[i, j] = val
                    parent[i, j] = suffix_parent[i][km]
                    if val > best_len:
                        best_len = val
                        best_end = (i, j)
        
        # Check time periodically
        if j % 50 == 0 and time.time() > deadline:
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
