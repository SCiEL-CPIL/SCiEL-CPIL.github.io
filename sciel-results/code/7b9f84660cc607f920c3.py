import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [primes[0]]
    
    # dp[i][j] = length of longest valid sequence ending with primes[i], primes[j] (j < i)
    # prev[i][j] = index k such that sequence ended with primes[k], primes[j]
    # If no such k, prev[i][j] = -1 (sequence is just primes[j], primes[i])
    
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    global_best_len = 1
    global_best_i = 0
    global_best_j = -1
    
    # Max possible gap is 10000
    MAX_GAP = 10000
    
    # Iterate through the "middle" prime index k
    # We compute dp[i][k] for all i > k
    for k in range(N):
        # Check time
        if time.time() > deadline:
            break
            
        # Build prefix max structure for k
        # We need to query: max dp[k][m] for P[k]-P[m] <= d
        # m < k
        
        # Arrays for prefix max
        # Index is gap value
        pref_len = [0] * (MAX_GAP + 1)
        pref_arg = [-1] * (MAX_GAP + 1)
        
        # Fill raw max for each exact gap
        # We can just update pref_len directly if we do it carefully?
        # No, we need max over all gaps <= d.
        # So we first collect max len for each exact gap, then prefix sum.
        
        # Optimization: Instead of clearing 10001 array every time,
        # we can just use a dict or sparse list if N is small, but N is ~1200.
        # 1200 * 10000 is fine.
        
        for m in range(k):
            gap = primes[k] - primes[m]
            length = dp[k][m]
            if length > pref_len[gap]:
                pref_len[gap] = length
                pref_arg[gap] = m
        
        # Compute prefix max
        cur_max = 0
        cur_arg = -1
        for g in range(1, MAX_GAP + 1):
            if pref_len[g] > cur_max:
                cur_max = pref_len[g]
                cur_arg = pref_arg[g]
            pref_len[g] = cur_max
            pref_arg[g] = cur_arg
        
        # Now compute dp[i][k] for all i > k
        for i in range(k + 1, N):
            gap = primes[i] - primes[k]
            # gap is always <= 10000
            
            if pref_len[gap] > 0:
                dp[i][k] = 1 + pref_len[gap]
                prev[i][k] = pref_arg[gap]
            else:
                dp[i][k] = 2
                prev[i][k] = -1
            
            if dp[i][k] > global_best_len:
                global_best_len = dp[i][k]
                global_best_i = i
                global_best_j = k
    
    # Reconstruct path
    if global_best_len == 1:
        return [primes[0]]
        
    path = []
    i = global_best_i
    j = global_best_j
    
    while j != -1:
        path.append(primes[i])
        i, j = j, prev[i][j]
    path.append(primes[i]) # The first prime in the sequence
    
    path.reverse()
    return path
