"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    import time
    
    start_time = time.time()
    limit = 10000
    
    # Generate primes up to 10000 using Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [primes[0]]
    
    # dp[i] stores the length of the longest valid subsequence ending at primes[i]
    # prev[i] stores the index of the previous element in that subsequence
    dp = [1] * n
    prev = [-1] * n
    
    # Iterate through primes
    for i in range(1, n):
        pi = primes[i]
        best_len = 1
        best_prev = -1
        
        # Check all possible predecessors j < i
        for j in range(i):
            pj = primes[j]
            gap = pi - pj
            
            # If the subsequence ending at j has length 1, we can always extend it
            if dp[j] == 1:
                if dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
            else:
                # Check if the gap is non-decreasing
                # The previous gap in the subsequence ending at j is primes[j] - primes[prev[j]]
                prev_gap = pj - primes[prev[j]]
                if gap >= prev_gap:
                    if dp[j] + 1 > best_len:
                        best_len = dp[j] + 1
                        best_prev = j
                        
        dp[i] = best_len
        prev[i] = best_prev
        
        # Time check (safety margin, though O(N^2) for N=1229 is very fast)
        if time.time() - start_time > 80:
            break

    # Find the index with the maximum length
    max_len = max(dp)
    end_idx = dp.index(max_len)
    
    # Backtrack to reconstruct the sequence
    seq = []
    curr = end_idx
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]
        
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
