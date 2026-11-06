import math
import bisect
import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    limit = 10000
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, p in enumerate(is_prime) if p]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    if N == 2:
        return primes
    
    # dp[i][j] = max length of valid sequence ending with primes[i], primes[j]
    # Use a list of arrays for better cache locality
    dp = [[0] * N for _ in range(N)]
    pred = [[-1] * N for _ in range(N)]
    
    # For each column j, we need suffix max of dp[:, j]
    # Process j from 1 to N-1
    for j in range(1, N):
        # Build suffix max for column j (dp[k][j] for k=0..j-1)
        # But we need this BEFORE computing dp[:, j], so we can't do it here
        # Instead, we precompute for each i, the suffix max over k in [k_min, i-1] of dp[k][i]
        
        # For each i < j, compute dp[i][j]
        for i in range(j):
            gap = primes[j] - primes[i]
            target = primes[i] - gap
            k_min = bisect.bisect_left(primes, target, 0, i)
            
            if k_min < i:
                # Need max of dp[k][i] for k in [k_min, i-1]
                # This was precomputed as suf_max[i][k_min]
                # But we need to build it before using it
                # Let's restructure: build suf_max for column i first
                pass
        # This approach has a dependency issue. Let me restructure.
        break
    
    # Restructured: process by "last prime" j, and for each j, we need
    # for each i < j: max of dp[k][i] for k in [k_min(i,j), i-1]
    # dp[k][i] depends on k < i, so when processing j, all dp[k][i] for k < i < j are known.
    # So we can precompute suffix max arrays for each column i as we go.
    
    # suf_max[i][k] = max(dp[t][i]) for t in [k, i-1]
    # We build suf_max[i] after finishing all dp[t][i] for t < i
    # But dp[t][i] is computed when we process column i (as j=i)
    
    # Let me redo this properly:
    # Process j from 1 to N-1:
    #   For each i in [0, j-1]:
    #     Compute dp[i][j] using suf_max[i][k_min]
    #   After computing all dp[i][j] for i < j, build suf_max[j] for future use
    
    # But wait: dp[i][j] uses suf_max[i][k_min], which is max of dp[k][i] for k in [k_min, i-1]
    # dp[k][i] was computed when j was i (earlier iteration)
    # So suf_max[i] must be built after processing column i
    
    # Corrected approach:
    # Initialize: for all pairs (i,j) with j=i+1, dp[i][j] = 2 (no predecessor)
    # Actually, let's just do it iteratively:
    
    # suf_max[i] is an array of length i: suf_max[i][k] = max(dp[t][i]) for t in [k, i-1]
    # suf_arg[i] is an array of length i: suf_arg[i][k] = argmax t in [k, i-1] of dp[t][i]
    
    suf_max = [None] * N
    suf_arg = [None] * N
    
    # Base case: for j=1, i=0: dp[0][1] = 2 (sequence [p0, p1])
    # Build suf_max[0] = empty (no predecessors for i=0)
    # Actually suf_max[i] has length i, so suf_max[0] has length 0
    
    for j in range(1, N):
        # First, compute dp[i][j] for all i < j
        for i in range(j):
            gap = primes[j] - primes[i]
            target = primes[i] - gap
            k_min = bisect.bisect_left(primes, target, 0, i)
            
            if k_min < i:
                # Use precomputed suf_max[i]
                best_len = suf_max[i][k_min]
                best_k = suf_arg[i][k_min]
                dp[i][j] = best_len + 1
                pred[i][j] = best_k
            else:
                dp[i][j] = 2
                pred[i][j] = -1
        
        # Now build suf_max[j] and suf_arg[j] for use in future iterations
        # suf_max[j][k] = max(dp[t][j]) for t in [k, j-1]
        sm = [0] * j
        sa = [-1] * j
        cur_max = 0
        cur_arg = -1
        for k in range(j - 1, -1, -1):
            val = dp[k][j]
            if val > cur_max:
                cur_max = val
                cur_arg = k
            sm[k] = cur_max
            sa[k] = cur_arg
        suf_max[j] = sm
        suf_arg[j] = sa
        
        # Time check
        if time.time() > deadline - 5.0:
            break
    
    # Find the maximum length
    max_len = 0
    end_i = 0
    end_j = 1
    for j in range(1, N):
        for i in range(j):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                end_i = i
                end_j = j
    
    # Reconstruct the path
    path = []
    i, j = end_i, end_j
    while j > 0:
        path.append(primes[j])
        k = pred[i][j]
        if k == -1:
            path.append(primes[i])
            break
        i, j = k, i
    
    path.reverse()
    return path
