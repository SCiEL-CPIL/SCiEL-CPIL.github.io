import numpy as np
import time

def construct():
    start_time = time.time()
    # Deadline guard: 50 seconds to be safe within 60s limit
    deadline = start_time + 50.0
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    M = len(primes)
    G = limit  # Max gap is 10000 - 2 = 9998, so size 10001 is safe
    
    if M == 0:
        return []
    if M == 1:
        return primes
        
    # 2. Initialize arrays
    # dp[j][g]: max length of subsequence ending at primes[j] with last gap exactly g
    dp = np.zeros((M, G + 1), dtype=np.int32)
    # prev_i[j][g]: index of previous prime in the optimal subsequence
    prev_i = np.full((M, G + 1), -1, dtype=np.int16)
    # prev_g[j][g]: last gap of the previous prime in the optimal subsequence
    prev_g = np.full((M, G + 1), -1, dtype=np.int16)
    # prefix_dp[j][g]: max length of subsequence ending at primes[j] with last gap <= g
    prefix_dp = np.zeros((M, G + 1), dtype=np.int32)
    # argmax_gap[j][g]: the gap k <= g that achieves prefix_dp[j][g]
    argmax_gap = np.zeros((M, G + 1), dtype=np.int16)
    
    # Base case: A subsequence of length 1 has no gap.
    # We represent this by setting dp[j][0] = 1.
    # This ensures that any transition to a new prime will yield at least length 2.
    for j in range(M):
        dp[j][0] = 1
        prefix_dp[j][0] = 1
        argmax_gap[j][0] = 0
        # For g > 0, prefix_dp[j][g] will be 1 initially if no other updates occur
        # But we compute prefix_dp after all updates for row j.
        # So we just initialize dp[j][0]=1 here.
        
    # 3. DP Iteration
    for j in range(1, M):
        # Check time
        if time.time() > deadline:
            break
            
        pj = primes[j]
        for i in range(j):
            g = pj - primes[i]
            # g is the gap between primes[i] and primes[j]
            # We can extend any valid subsequence ending at primes[i] with last gap <= g
            cand = 1 + prefix_dp[i][g]
            if cand > dp[j][g]:
                dp[j][g] = cand
                prev_i[j][g] = i
                prev_g[j][g] = argmax_gap[i][g]
                
        # Compute prefix max and argmax for row j
        # We do this with a Python loop for clarity and to handle argmax tracking
        # G is 10000, M is ~1229, so 12M iterations total.
        max_val = 0
        max_idx = 0
        for k in range(G + 1):
            if dp[j][k] > max_val:
                max_val = dp[j][k]
                max_idx = k
            prefix_dp[j][k] = max_val
            argmax_gap[j][k] = max_idx
            
    # 4. Find the global maximum length and corresponding state
    max_len = 0
    best_j = 0
    best_g = 0
    for j in range(M):
        # We can optimize this search, but M*G is 12M, which is fine
        for g in range(G + 1):
            if dp[j][g] > max_len:
                max_len = dp[j][g]
                best_j = j
                best_g = g
                
    # 5. Backtrack to reconstruct the sequence
    path = []
    cur_j = best_j
    cur_g = best_g
    while cur_j != -1:
        path.append(primes[cur_j])
        if prev_i[cur_j][cur_g] == -1:
            break
        next_j = prev_i[cur_j][cur_g]
        next_g = prev_g[cur_j][cur_g]
        cur_j = next_j
        cur_g = next_g
        
    path.reverse()
    return path
