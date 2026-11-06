def construct():
    from math import isqrt
    
    # Generate primes up to 10000
    sieve = [True] * 10001
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(10000) + 1):
        if sieve[i]:
            for j in range(i*i, 10001, i):
                sieve[j] = False
    primes = [i for i in range(2, 10001) if sieve[i]]
    n = len(primes)
    
    # dp[j] = (length, last_gap, prev_index) for best sequence ending at primes[j]
    # We want to maximize length, then minimize last_gap
    dp_len = [1] * n
    dp_last_gap = [0] * n  # 0 means no gap (single element)
    dp_prev = [-1] * n
    
    for j in range(1, n):
        best_len = 1
        best_last_gap = 0
        best_prev = -1
        for i in range(j):
            gap = primes[j] - primes[i]
            if dp_last_gap[i] == 0:
                # Sequence ending at i is just [primes[i]], can extend
                new_len = dp_len[i] + 1
                new_last_gap = gap
                if new_len > best_len or (new_len == best_len and new_last_gap < best_last_gap):
                    best_len = new_len
                    best_last_gap = new_last_gap
                    best_prev = i
            else:
                if gap >= dp_last_gap[i]:
                    new_len = dp_len[i] + 1
                    new_last_gap = gap
                    if new_len > best_len or (new_len == best_len and new_last_gap < best_last_gap):
                        best_len = new_len
                        best_last_gap = new_last_gap
                        best_prev = i
        dp_len[j] = best_len
        dp_last_gap[j] = best_last_gap
        dp_prev[j] = best_prev
    
    # Find the best ending
    best_end = 0
    for j in range(n):
        if dp_len[j] > dp_len[best_end] or (dp_len[j] == dp_len[best_end] and dp_last_gap[j] < dp_last_gap[best_end]):
            best_end = j
    
    # Reconstruct
    seq = []
    j = best_end
    while j != -1:
        seq.append(primes[j])
        j = dp_prev[j]
    seq.reverse()
    
    return seq
