import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0

    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # dp[i] is a list of (last_gap, length, prev_idx, prev_state_idx)
    # prev_idx is the index j in primes
    # prev_state_idx is the index of the state in dp[j]
    
    dp = []
    
    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        candidates = []
        # Base case: sequence of length 1 ending at primes[i]
        # last_gap = 0, length = 1, no prev
        candidates.append((0, 1, -1, -1))
        
        # Extend from previous primes
        for j in range(i):
            gap = primes[i] - primes[j]
            # For each state in dp[j], check if we can extend
            for k, (g_j, L_j, _, _) in enumerate(dp[j]):
                if gap >= g_j:
                    # New state for i
                    candidates.append((gap, L_j + 1, j, k))
        
        # Prune candidates
        # Sort by gap ascending
        candidates.sort(key=lambda x: x[0])
        
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            gap, length, pj, ps = c
            if length > max_len_so_far:
                pruned.append(c)
                max_len_so_far = length
            # If length <= max_len_so_far, it is dominated by a previous state 
            # with smaller gap and >= length
        
        dp.append(pruned)
        
        # Optimization: if time is running out, we can stop adding new primes 
        # and just return the best found so far. 
        # But we need to finish the loop to have dp for the last prime.
        # Given N=1229, the loop is fast.
        
    # Find the best sequence
    best_len = 0
    best_i = -1
    best_k = -1
    
    for i in range(len(dp)):
        for k, (gap, length, pj, ps) in enumerate(dp[i]):
            if length > best_len:
                best_len = length
                best_i = i
                best_k = k
                
    if best_i == -1:
        return []
        
    # Reconstruct
    seq_indices = []
    curr_i = best_i
    curr_k = best_k
    while curr_i != -1:
        seq_indices.append(curr_i)
        gap, length, pj, ps = dp[curr_i][curr_k]
        curr_i = pj
        curr_k = ps
        
    seq_indices.reverse()
    result = [primes[idx] for idx in seq_indices]
    
    return result
