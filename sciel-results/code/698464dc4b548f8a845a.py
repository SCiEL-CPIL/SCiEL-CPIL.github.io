import numpy as np
import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = start_time + 85.0

    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes

    # Precompute gaps? No, compute on the fly.
    
    # Arrays to store DP states
    # pref_max_arrs[i] is an array of size LIMIT+1
    # pref_max_arrs[i][g] = max length of sequence ending at primes[i] with last gap <= g
    pref_max_arrs = []
    argmax_gap_arrs = []
    parent_arrs = []
    prev_gap_arrs = []
    
    best_len_global = 1
    best_end_i = 0
    best_end_gap = 0
    
    for i in range(n):
        # Initialize exact_dp for current prime i
        # exact_dp[g] = max length of sequence ending at primes[i] with EXACT last gap g
        # Initialized to 1 (sequence [primes[i]])
        exact_dp = np.ones(LIMIT + 1, dtype=np.int32)
        par = np.full(LIMIT + 1, -1, dtype=np.int32)
        pgap = np.zeros(LIMIT + 1, dtype=np.int32)
        
        # Check time
        if time.time() > DEADLINE:
            break

        # Iterate over previous primes j
        for j in range(i):
            gap = primes[i] - primes[j]
            # Query best length ending at j with last gap <= gap
            len_prev = pref_max_arrs[j][gap]
            len_curr = 1 + len_prev
            
            if len_curr > exact_dp[gap]:
                exact_dp[gap] = len_curr
                par[gap] = j
                # The gap used at j is the one that achieved the max
                pgap[gap] = argmax_gap_arrs[j][gap]
        
        # Build prefix max and argmax for current i
        # pref_max_i[g] = max(exact_dp[0..g])
        pref_max_i = np.maximum.accumulate(exact_dp)
        
        # argmax_gap_i[g] = index h in [0..g] such that exact_dp[h] is max
        argmax_gap_i = np.zeros(LIMIT + 1, dtype=np.int32)
        best_g = 0
        best_v = exact_dp[0]
        argmax_gap_i[0] = 0
        for g in range(1, LIMIT + 1):
            if exact_dp[g] > best_v:
                best_v = exact_dp[g]
                best_g = g
            argmax_gap_i[g] = best_g
        
        # Update global best
        max_val = pref_max_i[LIMIT]
        if max_val > best_len_global:
            best_len_global = max_val
            best_end_i = i
            best_end_gap = best_g # The gap that achieved the max for this i
        
        # Store arrays for future use
        pref_max_arrs.append(pref_max_i)
        argmax_gap_arrs.append(argmax_gap_i)
        parent_arrs.append(par)
        prev_gap_arrs.append(pgap)
        
        if time.time() > DEADLINE:
            break

    # Reconstruct sequence
    if best_len_global == 1:
        return [primes[0]]
        
    seq = []
    i = best_end_i
    g = best_end_gap
    
    while i >= 0:
        seq.append(primes[i])
        if i == 0:
            break
        j = parent_arrs[i][g]
        g_prev = prev_gap_arrs[i][g]
        i = j
        g = g_prev
        
    seq.reverse()
    return seq
