def construct():
    N = 123457
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)  # largest prime factor
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
                # Actually, we want the LARGEST prime factor, so we should update
                # Let me redo: initialize to 1, then for each prime, mark multiples
    # Better approach: initialize lpf to 1, then for each prime p, for each multiple, set lpf[multiple] = max(lpf[multiple], p)
    # But that's O(n log log n * something). Let me do it differently.
    
    # Actually, the standard way: for each number, its largest prime factor can be found by
    # iterating primes and updating. Let me use a different sieve.
    
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # Since we iterate i in increasing order, the last prime that divides j is the largest
    
    # Now search for a in [1, N-1] minimizing max(lpf[a], lpf[N-a])
    best_val = float('inf')
    best_a = 1
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
        # Early termination: if best_val is 1, we can't do better
        if best_val == 1:
            break
    
    b = N - best_a
    return [best_a, b]
