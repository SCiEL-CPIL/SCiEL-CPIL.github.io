import time
import math
from bisect import bisect_left

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
    # We use a list of lists for dp and parent
    dp = [[2] * N for _ in range(N)]
    parent = [[-1] * N for _ in range(N)]
    
    # suffix_max[i][k] = max(dp[k][i], ..., dp[i-1][i])
    # suffix_parent[i][k] = argmax index
    suffix_max = [None] * N
    suffix_parent = [None] * N
    
    best_len = 2
    best_end = (0, 1)
    
    # j is the index of the second prime in the pair
    for j in range(1, N):
        # Compute suffix_max for i = j-1
        # Because dp[k][j-1] for k < j-1 are now finalized
        i = j - 1
        if i > 0:
            sm = [0] * i
            sp = [-1] * i
            sm[i-1] = dp[i-1][i]
            sp[i-1] = i-1
            for k in range(i-2, -1, -1):
                if dp[k][i] > sm[k+1]:
                    sm[k] = dp[k][i]
                    sp[k] = k
                else:
                    sm[k] = sm[k+1]
                    sp[k] = sp[k+1]
            suffix_max[i] = sm
            suffix_parent[i] = sp
        else:
            suffix_max[i] = []
            suffix_parent[i] = []
            
        # Now compute dp[i][j] for all i < j
        time_check_interval = 50
        for i in range(j):
            # Condition: p[k] >= 2*p[i] - p[j], find smallest valid k in [0, i-1]
            threshold = 2 * primes[i] - primes[j]
            # Use bisect to find first prime >= threshold
            km = bisect_left(primes, threshold, 0, i)
            
            if km < i:
                # Valid range is [km, i-1]
                # Max value is suffix_max[i][km]
                val = 1 + suffix_max[i][km]
                if val > dp[i][j]:
                    dp[i][j] = val
                    parent[i][j] = suffix_parent[i][km]
                    if val > best_len:
                        best_len = val
                        best_end = (i, j)
            # else dp[i][j] remains 2
            
            # Check time periodically
            if (i + j) % time_check_interval == 0 and time.time() > deadline:
                # Reconstruct and return
                result = []
                ci, cj = best_end
                result.append(primes[cj])
                result.append(primes[ci])
                while parent[ci][cj] != -1:
                    ck = parent[ci][cj]
                    result.append(primes[ck])
                    cj = ci
                    ci = ck
                result.reverse()
                return result
                
    # Reconstruct the sequence
    result = []
    ci, cj = best_end
    result.append(primes[cj])
    result.append(primes[ci])
    while parent[ci][cj] != -1:
        ck = parent[ci][cj]
        result.append(primes[ck])
        cj = ci
        ci = ck
        
    result.reverse()
    return result
