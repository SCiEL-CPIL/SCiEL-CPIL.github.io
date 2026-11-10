import time
from bisect import bisect_left
from math import isqrt

def construct():
    """
    Finds the longest increasing subsequence of primes <= 10000 
    such that the gaps between consecutive primes are non-decreasing.
    """
    deadline = time.time() + 85  # Safety margin
    
    # 1. Sieve of Eratosthenes to get primes up to 10000
    n_max = 10000
    is_prime = [True] * (n_max + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(n_max) + 1):
        if is_prime[i]:
            for j in range(i*i, n_max + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n_max + 1) if is_prime[i]]
    m = len(primes)
    
    # dp[i][j] = max length of valid chain ending at primes[i], primes[j] (i < j)
    # We only need dp[k][i] for k < i to compute dp[i][j].
    # We store the full dp table to allow reconstruction, but we can optimize memory
    # if we only stored the back pointers. However, m is small (~1229), so 
    # m*m ~ 1.5M entries is fine for memory.
    
    # amax[i][j] stores the k that gave the max length for dp[i][j]
    # If no predecessor, amax[i][j] = -1
    
    dp = [[0] * m for _ in range(m)]
    amax = [[-1] * m for _ in range(m)]
    
    # S_val[j][k] = max(dp[x][j] for x in [k, j-1])
    # S_arg[j][k] = argmax x
    # We compute these for each column j after finishing all i < j for that j.
    S_val = [None] * m
    S_arg = [None] * m
    
    best_len = 2
    best_i = 0
    best_j = 1
    
    for j in range(m):
        if time.time() > deadline:
            break
        bj = primes[j]
        for i in range(j):
            if time.time() > deadline:
                break
            bi = primes[i]
            # Condition for predecessor k < i:
            # primes[i] - primes[k] <= primes[j] - primes[i]
            # primes[k] >= 2*primes[i] - primes[j]
            T = 2 * bi - bj
            
            # Find smallest k in [0, i) such that primes[k] >= T
            # bisect_left returns the index where T would be inserted to maintain order
            # i.e., the first index with value >= T
            k_min = bisect_left(primes, T, 0, i)
            
            if k_min < i:
                # There is at least one valid k
                # We need max dp[x][i] for x in [k_min, i-1]
                # This is stored in S_val[i][k_min]
                max_prev = S_val[i][k_min]
                dp[i][j] = 2 + max_prev
                amax[i][j] = S_arg[i][k_min]
                
                if dp[i][j] > best_len:
                    best_len = dp[i][j]
                    best_i = i
                    best_j = j
            else:
                # No valid predecessor, chain starts at i, j
                dp[i][j] = 2
                amax[i][j] = -1
                # Length 2 is not better than current best_len which is at least 2
                # So no update needed for best_len unless best_len < 2, but it's initialized to 2.
                
        # Compute suffix max for column j
        # S_val[j][k] for k in [0, j-1]
        if j > 0:
            sv = [0] * j
            sa = [0] * j
            # k = j-1
            sv[j-1] = dp[j-1][j]
            sa[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[k][j] >= sv[k+1]:
                    sv[k] = dp[k][j]
                    sa[k] = k
                else:
                    sv[k] = sv[k+1]
                    sa[k] = sa[k+1]
            S_val[j] = sv
            S_arg[j] = sa
    
    # Reconstruct the chain
    if best_len <= 2:
        return [primes[0], primes[1]]
        
    chain = [primes[best_j], primes[best_i]]
    ci, cj = best_i, best_j
    clen = best_len
    
    while clen > 2:
        k = amax[ci][cj]
        if k < 0:
            break
        chain.append(primes[k])
        cj = ci
        ci = k
        clen -= 1
        
    chain.reverse()
    
    # Verify and potentially extend the chain by checking if we can add more primes
    # This is a simple post-processing step to see if we can extend the found chain
    # by checking if we can prepend or append more primes that maintain the property
    
    # Try to extend backwards (prepend primes)
    if len(chain) > 1:
        while True:
            first_two_gaps = chain[1] - chain[0]
            # We need a prime p < chain[0] such that chain[0] - p <= first_two_gaps
            # i.e., p >= chain[0] - first_two_gaps
            min_p = chain[0] - first_two_gaps
            if min_p < 2:
                break
            # Find the largest prime <= chain[0] that is >= min_p
            # We can search backwards in primes list
            idx = bisect_left(primes, min_p)
            if idx < len(primes) and primes[idx] < chain[0]:
                # We can add primes[idx]
                chain.insert(0, primes[idx])
            else:
                break
    
    # Try to extend forwards (append primes)
    if len(chain) > 1:
        while True:
            last_two_gaps = chain[-1] - chain[-2]
            # We need a prime p > chain[-1] such that p - chain[-1] >= last_two_gaps
            # i.e., p >= chain[-1] + last_two_gaps
            max_p = chain[-1] + last_two_gaps
            if max_p > 10000:
                break
            # Find the smallest prime >= max_p
            idx = bisect_left(primes, max_p)
            if idx < len(primes) and primes[idx] <= 10000:
                # We can add primes[idx]
                chain.append(primes[idx])
            else:
                break
    
    return chain
