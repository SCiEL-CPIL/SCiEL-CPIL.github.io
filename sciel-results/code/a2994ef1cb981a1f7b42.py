import time
from math import isqrt
from bisect import bisect_left

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
    
    # dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # prev[i][j] = index h such that dp[h][i] was used to compute dp[i][j]
    # If no such h, prev[i][j] = -1
    
    # Initialize
    dp = [[0] * K for _ in range(K)]
    prev = [[-1] * K for _ in range(K)]
    
    # We only care about i < j
    # Base case: any pair (i, j) with i < j has length at least 2.
    # We will compute dp[i][j] for i < j.
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    for i in range(K - 1):
        if time.time() > deadline:
            break
            
        # Build suffix max and argmax for dp[h][i] for h in [0, i-1]
        # We need to query max dp[h][i] for h in [H_min, i-1]
        # Let's create arrays for this specific i
        if i > 0:
            max_vals = [0] * i
            max_idxs = [0] * i
            max_vals[i-1] = dp[i-1][i]
            max_idxs[i-1] = i-1
            for h in range(i-2, -1, -1):
                if dp[h][i] > max_vals[h+1]:
                    max_vals[h] = dp[h][i]
                    max_idxs[h] = h
                else:
                    max_vals[h] = max_vals[h+1]
                    max_idxs[h] = max_idxs[h+1]
        else:
            max_vals = []
            max_idxs = []
            
        for j in range(i + 1, K):
            gap = primes[j] - primes[i]
            # Find h < i such that primes[i] - primes[h] <= gap
            # primes[h] >= primes[i] - gap
            target = primes[i] - gap
            h_min = bisect_left(primes, target)
            
            # We need h < i
            if h_min >= i:
                # No valid h
                dp[i][j] = 2
                prev[i][j] = -1
            else:
                # Query max in [h_min, i-1]
                # Our max_vals array is indexed 0 to i-1
                # max_vals[k] holds max for range [k, i-1]
                # So we just access max_vals[h_min]
                best_h = max_idxs[h_min]
                best_len_from_h = max_vals[h_min]
                
                dp[i][j] = best_len_from_h + 1
                prev[i][j] = best_h
                
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_i = i
                    best_j = j
                    
    # Backtrack to find the sequence
    seq = [primes[best_j]]
    i = best_i
    j = best_j
    while i != -1:
        seq.append(primes[i])
        h = prev[i][j]
        if h == -1:
            break
        j = i
        i = h
        
    seq.reverse()
    
    return seq
