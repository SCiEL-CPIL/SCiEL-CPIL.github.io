def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for 90s limit with safety margin
    limit = 1000000
    N = limit + 1
    
    # Compute largest prime factor for all numbers up to N using sieve
    # This is O(N log log N) and should complete well within the time limit
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    # Find best n by scanning all candidates
    best_val = float('inf')
    best_n = 10000
    
    for n in range(10000, limit + 1):
        if time.time() > deadline:
            break
        v = lpf[n] if lpf[n] >= lpf[n+1] else lpf[n+1]
        if v < best_val:
            best_val = v
            best_n = n
    
    return best_n
