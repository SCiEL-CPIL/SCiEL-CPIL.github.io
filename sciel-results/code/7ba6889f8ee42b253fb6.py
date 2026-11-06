def construct():
    from math import isqrt
    import time

    start_time = time.time()
    deadline = 90.0

    # Generate primes up to 10000
    sieve = [True] * 10001
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(10000) + 1):
        if sieve[i]:
            for j in range(i*i, 10001, i):
                sieve[j] = False
    primes = [i for i in range(2, 10001) if sieve[i]]
    n = len(primes)

    # dp[j] is a list of (length, last_gap, prev_index)
    dp = [[] for _ in range(n)]

    for j in range(n):
        # best_gap_for_len: {length: (min_gap, prev_index)}
        best_gap_for_len = {1: (0, -1)}
        
        for i in range(j):
            gap_base = primes[j] - primes[i]
            for L, G, prev in dp[i]:
                if gap_base >= G:
                    new_L = L + 1
                    new_G = gap_base
                    if new_L not in best_gap_for_len:
                        best_gap_for_len[new_L] = (new_G, i)
                    else:
                        existing_G, existing_prev = best_gap_for_len[new_L]
                        if new_G < existing_G:
                            best_gap_for_len[new_L] = (new_G, i)
        
        # Build frontier
        lengths = sorted(best_gap_for_len.keys(), reverse=True)
        frontier = []
        min_gap_so_far = float('inf')
        
        for L in lengths:
            G, prev = best_gap_for_len[L]
            if G < min_gap_so_far:
                frontier.append((L, G, prev))
                min_gap_so_far = G
        
        dp[j] = frontier

    # Find the best state across all j
    best_len = 0
    best_gap = float('inf')
    best_j = -1
    for j in range(n):
        for L, G, prev in dp[j]:
            if L > best_len or (L == best_len and G < best_gap):
                best_len = L
                best_gap = G
                best_j = j
                
    # Reconstruct
    seq = []
    j = best_j
    while j != -1:
        seq.append(primes[j])
        target_L = len(seq)
        prev_idx = -1
        for L, G, prev in dp[j]:
            if L == target_L:
                prev_idx = prev
                break
        j = prev_idx
        
    seq.reverse()
    return seq
