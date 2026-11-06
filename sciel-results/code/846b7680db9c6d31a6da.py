import numpy as np
import time

def construct():
    start_time = time.time()
    
    # 1. Generate Primes
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
        
    max_gap = limit
    
    # 2. Initialize Arrays
    # dp[i][g]: max length ending at i with last gap exactly g
    dp = np.zeros((N, max_gap + 1), dtype=np.int32)
    # prev_idx[i][g]: index of previous prime
    prev_idx = np.zeros((N, max_gap + 1), dtype=np.int32)
    # prev_gap[i][g]: gap of previous prime
    prev_gap = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    # prefix_max_len[i][g]: max length ending at i with last gap <= g
    prefix_max_len = np.zeros((N, max_gap + 1), dtype=np.int32)
    # prefix_max_gap[i][g]: gap value that achieved prefix_max_len[i][g]
    prefix_max_gap = np.zeros((N, max_gap + 1), dtype=np.int32)
    
    # 3. DP
    for i in range(N):
        # Base case: sequence of length 1, gap 0
        dp[i][0] = 1
        prefix_max_len[i][0] = 1
        prefix_max_gap[i][0] = 0
        
        for k in range(i):
            g = primes[i] - primes[k]
            # Get best previous state
            best_len = prefix_max_len[k][g]
            best_g_prev = prefix_max_gap[k][g]
            
            if best_len > 0:
                new_len = best_len + 1
                if new_len > dp[i][g]:
                    dp[i][g] = new_len
                    prev_idx[i][g] = k
                    prev_gap[i][g] = best_g_prev
        
        # Compute prefix max for row i
        # We need to do this AFTER processing all k?
        # No, dp[i][g] is updated for various g.
        # We compute prefix max over the final dp[i] row.
        current_max = 0
        current_max_gap = 0
        for g in range(1, max_gap + 1):
            if dp[i][g] > current_max:
                current_max = dp[i][g]
                current_max_gap = g
            else:
                current_max = max(current_max, dp[i][g]) # already handled by logic above?
                # Actually:
                if dp[i][g] > current_max:
                    current_max = dp[i][g]
                    current_max_gap = g
            prefix_max_len[i][g] = current_max
            prefix_max_gap[i][g] = current_max_gap

    # 4. Find Best
    # Flatten dp to find global max
    # Note: dp[i][0] is 1 for all i.
    max_val = 0
    best_i = 0
    best_g = 0
    for i in range(N):
        row_max = np.max(dp[i])
        if row_max > max_val:
            max_val = row_max
            best_i = i
            best_g = np.argmax(dp[i])
            
    # 5. Backtrack
    seq = []
    cur_i = best_i
    cur_g = best_g
    while True:
        seq.append(primes[cur_i])
        if cur_g == 0:
            break
        k = prev_idx[cur_i][cur_g]
        g_prev = prev_gap[cur_i][cur_g]
        cur_i = k
        cur_g = g_prev
        
    seq.reverse()
    return [int(x) for x in seq]
