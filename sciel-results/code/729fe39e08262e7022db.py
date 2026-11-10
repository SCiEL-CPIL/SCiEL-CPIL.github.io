def construct():
    N = 123457
    
    # Sieve for Largest Prime Factor (LPF)
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    # P(1) = 1
    lpf[1] = 1
    
    min_max_p = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We only need to check a up to N//2 due to symmetry
    for a in range(1, N // 2 + 1):
        b = N - a
        # P(a) is lpf[a], P(b) is lpf[b]
        current_max = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if current_max < min_max_p:
            min_max_p = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
