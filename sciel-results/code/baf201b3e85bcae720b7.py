import bisect
from math import isqrt

def construct():
    # Generate primes
    is_prime = [True] * 10001
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(10000) + 1):
        if is_prime[i]:
            for j in range(i*i, 10001, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N < 2:
        return primes
        
    # dp[i][j] = max length ending at primes[i], primes[j]
    # prev[i][j] = index of prime before primes[i] in the sequence
    dp = [[2] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    best_len = 2
    best_j, best_k = 0, 1
    
    for j in range(N - 1):
        # Build suffix max arrays for dp[0..j-1][j]
        # We need max dp[i][j] for i in [L, j-1]
        # suffix_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        # suffix_idx[i] = argmax(...)
        
        # We only need indices < j
        # Initialize with -inf or similar, but we know values are >= 2
        # We can just build lists of size j
        suffix_val = [0] * j
        suffix_idx = [0] * j
        
        if j > 0:
            suffix_val[j-1] = dp[j-1][j]
            suffix_idx[j-1] = j-1
            for i in range(j-2, -1, -1):
                if dp[i][j] >= suffix_val[i+1]:
                    suffix_val[i] = dp[i][j]
                    suffix_idx[i] = i
                else:
                    suffix_val[i] = suffix_val[i+1]
                    suffix_idx[i] = suffix_idx[i+1]
        
        for k in range(j + 1, N):
            limit = 2 * primes[j] - primes[k]
            # Find smallest index L such that primes[L] >= limit
            # L must be < j to be valid
            L = bisect.bisect_left(primes, limit, 0, j)
            
            if L < j:
                val = suffix_val[L]
                idx = suffix_idx[L]
                new_len = val + 1
                if new_len > dp[j][k]:
                    dp[j][k] = new_len
                    prev[j][k] = idx
                    if new_len > best_len:
                        best_len = new_len
                        best_j = j
                        best_k = k
            # else: no valid i, dp[j][k] remains 2 (or whatever initialized)
            
    # Backtrack
    res = []
    j, k = best_j, best_k
    while k != -1:
        res.append(primes[k])
        j, k = k, prev[j][k]
        
    res.reverse()
    return res
