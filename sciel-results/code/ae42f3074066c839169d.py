import numpy as np
import time

def construct():
    start_time = time.time()
    # Deadline guard: 55 seconds to be safe within 60s limit
    deadline = start_time + 55.0
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int32)
    M = len(primes)
    G = limit  # Max gap is 10000 - 2 = 9998, so size 10001 is safe
    
    if M == 0:
        return []
    if M == 1:
        return [int(primes[0])]
        
    # 2. Initialize arrays
    # dp[j][g]: max length of subsequence ending at primes[j] with last gap exactly g
    dp = np.zeros((M, G + 1), dtype=np.int32)
    # prev_i[j][g]: index of previous prime in the optimal subsequence
    prev_i = np.full((M, G + 1), -1, dtype=np.int32)
    # prev_g[j][g]: last gap of the previous prime in the optimal subsequence
    prev_g = np.full((M, G + 1), -1, dtype=np.int32)
    # prefix_dp[j][g]: max length of subsequence ending at primes[j] with last gap <= g
    prefix_dp = np.zeros((M, G + 1), dtype=np.int32)
    # argmax_gap[j][g]: the gap k <= g that achieves prefix_dp[j][g]
    argmax_gap = np.zeros((M, G + 1), dtype=np.int32)
    
    # Base case: A subsequence of length 1 has no gap.
    # We represent this by setting dp[j][0] = 1.
    # This ensures that any transition to a new prime will yield at least length 2.
    for j in range(M):
        dp[j][0] = 1
        prefix_dp[j][0] = 1
        argmax_gap[j][0] = 0
        
    # 3. DP Iteration - Vectorized approach
    for j in range(1, M):
        # Check time
        if time.time() > deadline:
            break
            
        pj = primes[j]
        # For each previous prime i, compute gap g = pj - primes[i]
        # We want to find for each gap g, the best candidate
        # cand = 1 + prefix_dp[i][g] for all i where pj - primes[i] = g
        
        # Compute all gaps from primes[i] to pj for i < j
        gaps = pj - primes[:j]  # shape (j,)
        
        # For each gap value, find the best prefix_dp[i][g] where gap == g
        # We'll iterate over unique gaps
        unique_gaps = np.unique(gaps)
        
        for g in unique_gaps:
            # Find all i where gaps[i] == g
            mask = (gaps == g)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            # Get prefix_dp values for these indices at gap g
            vals = prefix_dp[indices, g]
            best_val = np.max(vals)
            best_idx_in_indices = np.argmax(vals)
            best_i = indices[best_idx_in_indices]
            
            cand = 1 + best_val
            if cand > dp[j, g]:
                dp[j, g] = cand
                prev_i[j, g] = best_i
                prev_g[j, g] = argmax_gap[best_i, g]
                
        # Compute prefix max and argmax for row j using NumPy
        row = dp[j, :]
        # Compute running max
        prefix_max = np.maximum.accumulate(row)
        prefix_dp[j, :] = prefix_max
        
        # Compute argmax: for each position k, find the last position where max is achieved
        # We need the rightmost index where the running max is achieved
        # Actually, we want the index that achieves the prefix max at each position
        # For argmax tracking, we need: argmax_gap[j][g] = argmax of dp[j][0:g+1]
        # This is the index k <= g that maximizes dp[j][k]
        
        # Use a cumulative approach: track the argmax as we go
        # We'll compute this with a loop since NumPy doesn't have a direct "argmax up to here"
        # But we can optimize: since dp values are small integers, we can use a different approach
        
        # Actually, let's use a vectorized approach for argmax:
        # For each g, argmax_gap[j][g] = argmax(dp[j][0:g+1])
        # We can compute this by comparing dp[j][k] with the running max
        
        # Efficient approach: iterate through and track
        cur_max = 0
        cur_argmax = 0
        for k in range(G + 1):
            if row[k] > cur_max:
                cur_max = row[k]
                cur_argmax = k
            argmax_gap[j, k] = cur_argmax
            
    # 4. Find the global maximum length and corresponding state
    max_len = 0
    best_j = 0
    best_g = 0
    for j in range(M):
        row_max = np.max(dp[j, :])
        if row_max > max_len:
            max_len = row_max
            best_j = j
            best_g = np.argmax(dp[j, :])
            
    # 5. Backtrack to reconstruct the sequence
    path = []
    cur_j = best_j
    cur_g = best_g
    while cur_j != -1:
        path.append(int(primes[cur_j]))
        if prev_i[cur_j][cur_g] == -1:
            break
        next_j = int(prev_i[cur_j][cur_g])
        next_g = int(prev_g[cur_j][cur_g])
        cur_j = next_j
        cur_g = next_g
        
    path.reverse()
    return path
