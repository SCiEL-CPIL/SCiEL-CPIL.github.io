import bisect
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)

    if N < 2:
        return primes

    # dp[i][j] = length of longest sequence ending with primes[i], primes[j] (j < i)
    # prev[i][j] = index k such that sequence ended with primes[k], primes[j]
    # If prev[i][j] is -1, the sequence is just [primes[j], primes[i]]
    
    dp = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]

    # Iterate j (the previous prime in the pair)
    for j in range(N):
        if time.time() > deadline:
            break
            
        # Prepare suffix max for dp[j][k] where k < j
        # We need max_{k in [idx, j-1]} dp[j][k]
        # suf_val[k] = max(dp[j][k], ..., dp[j][j-1])
        # suf_idx[k] = argmax
        
        if j > 0:
            suf_val = [0] * j
            suf_idx = [0] * j
            suf_val[j-1] = dp[j][j-1]
            suf_idx[j-1] = j-1
            for k in range(j-2, -1, -1):
                if dp[j][k] >= suf_val[k+1]:
                    suf_val[k] = dp[j][k]
                    suf_idx[k] = k
                else:
                    suf_val[k] = suf_val[k+1]
                    suf_idx[k] = suf_idx[k+1]
        else:
            suf_val = []
            suf_idx = []

        # Compute dp[i][j] for all i > j
        for i in range(j+1, N):
            if time.time() > deadline:
                break
            
            # Condition: p_j - p_k <= p_i - p_j  =>  p_k >= 2*p_j - p_i
            threshold = 2 * primes[j] - primes[i]
            
            # Find smallest index idx such that primes[idx] >= threshold
            idx = bisect.bisect_left(primes, threshold)
            
            # We need k in [idx, j-1]
            if idx < j:
                # The max value is in suf_val[idx]
                best_val = suf_val[idx]
                best_k = suf_idx[idx]
                dp[i][j] = best_val + 1
                prev[i][j] = best_k
            else:
                # No valid k < j, so sequence length is 2
                dp[i][j] = 2
                prev[i][j] = -1

    # Find the maximum length and the ending pair (i, j)
    max_len = 0
    best_i = -1
    best_j = -1
    for i in range(N):
        for j in range(i):
            if dp[i][j] > max_len:
                max_len = dp[i][j]
                best_i = i
                best_j = j

    # Reconstruct the sequence
    if max_len == 0:
        return primes[:1] if primes else []

    # Reconstruct indices backwards
    # State (i, j) means sequence ends with p_j, p_i
    # prev[i][j] is k such that sequence ended with p_k, p_j
    
    indices = []
    i = best_i
    j = best_j
    indices.append(i)
    indices.append(j)
    
    while prev[i][j] != -1:
        k = prev[i][j]
        indices.append(k)
        i, j = j, k
        
    indices.reverse()
    
    # Map indices to prime values
    seq = [primes[idx] for idx in indices]
    return seq
