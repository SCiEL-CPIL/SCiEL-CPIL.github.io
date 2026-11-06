"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # Generate all primes <= 10000 using sieve
    is_prime = [True] * 10001
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(10000) + 1):
        if is_prime[i]:
            for j in range(i * i, 10001, i):
                is_prime[j] = False
    primes = [p for p in range(2, 10001) if is_prime[p]]
    n = len(primes)
    
    # dp[j][g] = (length, parent_idx, parent_gap)
    # State: longest valid prime subsequence ending at primes[j] with last gap g
    dp = [dict() for _ in range(n)]
    
    # Base case: each prime alone, length 1, last gap 0
    for j in range(n):
        dp[j][0] = (1, -1, 0)
    
    # Fill DP table
    for j in range(n):
        for i in range(j):
            gap = primes[j] - primes[i]
            for g, (length, _, _) in dp[i].items():
                if gap >= g:
                    new_length = length + 1
                    if gap not in dp[j] or new_length > dp[j][gap][0]:
                        dp[j][gap] = (new_length, i, g)
    
    # Find the best ending state
    max_len = 0
    best_j, best_g = -1, 0
    for j in range(n):
        for g, (length, _, _) in dp[j].items():
            if length > max_len:
                max_len = length
                best_j, best_g = j, g
    
    # Reconstruct the sequence by following parent pointers
    seq = []
    j, g = best_j, best_g
    while j != -1:
        seq.append(primes[j])
        _, parent, parent_g = dp[j][g]
        j, g = parent, parent_g
    
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
