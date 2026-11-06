import time
import math

def construct():
    deadline = time.time() + 78  # Safety margin
    
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
    
    # parent[i][j] stores the index k such that dp[k][i] was the predecessor for dp[i][j]
    parent = [[-1] * N for _ in range(N)]
    
    # smax[i][k] = max(dp[k][i], dp[k+1][i], ..., dp[i-1][i])
    # smax_idx[i][k] = index of the max value
    # We store these as lists of lists. smax[i] has length i.
    smax = [None] * N
    smax_idx = [None] * N
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    # dp_col will hold dp[i][j] for the current j
    for j in range(1, N):
        if time.time() > deadline:
            break
            
        k_ptr = 0
        dp_col = [2] * j  # Initialize with base case length 2
        
        for i in range(j):
            T = 2 * primes[i] - primes[j]
            
            # Find smallest k < i such that primes[k] >= T
            # k_ptr is non-decreasing as i increases for a fixed j
            while k_ptr < i and primes[k_ptr] < T:
                k_ptr += 1
                
            if k_ptr < i:
                # There is a valid predecessor
                # The best predecessor is the one that maximizes dp[k][i] for k >= k_ptr
                val = 1 + smax[i][k_ptr]
                idx = smax_idx[i][k_ptr]
                
                if val > dp_col[i]:
                    dp_col[i] = val
                    parent[i][j] = idx
                    if val > best_len:
                        best_len = val
                        best_i = i
                        best_j = j
            else:
                # No valid predecessor, length remains 2
                dp_col[i] = 2
                parent[i][j] = -1
        
        # Compute suffix maximums for column j to be used in future iterations
        # smax[j][k] = max(dp_col[k], dp_col[k+1], ..., dp_col[j-1])
        if j > 0:
            sm = [0] * j
            sm_idx = [-1] * j
            sm[j-1] = dp_col[j-1]
            sm_idx[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp_col[k] > sm[k+1]:
                    sm[k] = dp_col[k]
                    sm_idx[k] = k
                else:
                    sm[k] = sm[k+1]
                    sm_idx[k] = sm_idx[k+1]
            smax[j] = sm
            smax_idx[j] = sm_idx
            
    # Reconstruct the sequence
    if best_len <= 1:
        return primes[:best_len]
        
    result = []
    ci, cj = best_i, best_j
    result.append(primes[cj])
    result.append(primes[ci])
    
    while parent[ci][cj] != -1:
        ck = parent[ci][cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
        
    result.reverse()
    return result
