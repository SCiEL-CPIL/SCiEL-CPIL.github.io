import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 55.0

    MAX_P = 10000
    is_prime = np.ones(MAX_P + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(MAX_P)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = np.array([i for i, flag in enumerate(is_prime) if flag], dtype=np.int32)
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return [int(primes[0])]
        
    MAX_GAP = MAX_P
    
    dp = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    prev_i = np.full((N, MAX_GAP + 1), -1, dtype=np.int32)
    prev_g = np.full((N, MAX_GAP + 1), -1, dtype=np.int32)
    
    pref_max = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    pref_arg = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
    
    for j in range(N):
        dp[j, 0] = 1
        
    # Precompute prefix max/arg for j=0
    # dp[0] is 1 at index 0, 0 elsewhere
    # pref_max[0][g] should be 1 for all g >= 0
    pref_max[0, :] = 1
    pref_arg[0, :] = 0
    
    for j in range(1, N):
        if time.time() > deadline:
            break
            
        p_j = primes[j]
        
        for i in range(j):
            g = p_j - primes[i]
            if g > MAX_GAP:
                continue
                
            # Look up best value in row i for gap <= g
            val = pref_max[i, g]
            arg = pref_arg[i, g]
            
            if val + 1 > dp[j, g]:
                dp[j, g] = val + 1
                prev_i[j, g] = i
                prev_g[j, g] = arg
                
        # Update prefix max/arg for row j
        # We need max(dp[j][0..k]) for each k
        # np.maximum.accumulate does this
        pref_max[j, :] = np.maximum.accumulate(dp[j, :])
        # For argmax, we need the index of the max.
        # If there are ties, we want the one that allows future extensions?
        # Actually, any max is fine for length.
        # However, for reconstruction, we stored prev_g which is the arg.
        # If we pick a different arg with same value, it might lead to different sequence but same length.
        # np.argmax returns first max.
        # To match np.maximum.accumulate logic, we just need *a* valid arg.
        # We can compute argmax for each prefix? That's expensive.
        # Alternative:
        # We can store the arg that gave the max in pref_max.
        # Since np.maximum.accumulate is vectorized, getting arg is tricky.
        # But we can do:
        # current_max = 0
        # current_arg = -1
        # for k in range(MAX_GAP+1):
        #   if dp[j, k] > current_max:
        #     current_max = dp[j, k]
        #     current_arg = k
        #   pref_max[j, k] = current_max
        #   pref_arg[j, k] = current_arg
        # This is O(MAX_GAP) per j. N * MAX_GAP = 12M ops. Fast.
        
        current_max = 0
        current_arg = -1
        row_dp = dp[j, :]
        row_pmax = pref_max[j, :]
        row_parg = pref_arg[j, :]
        
        for k in range(MAX_GAP + 1):
            if row_dp[k] > current_max:
                current_max = row_dp[k]
                current_arg = k
            row_pmax[k] = current_max
            row_parg[k] = current_arg

    # Find global max
    max_len = np.max(dp)
    idx = np.unravel_index(np.argmax(dp), dp.shape)
    best_j = idx[0]
    best_g = idx[1]
    
    seq = [int(primes[best_j])]
    curr_j = best_j
    curr_g = best_g
    
    while prev_i[curr_j, curr_g] != -1:
        curr_i = prev_i[curr_j, curr_g]
        curr_g_prev = prev_g[curr_j, curr_g]
        seq.append(int(primes[curr_i]))
        curr_j = curr_i
        curr_g = curr_g_prev
        
    seq.reverse()
    
    return seq
