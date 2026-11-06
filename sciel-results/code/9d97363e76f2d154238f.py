def construct():
    N = 123457
    # Sieve for largest prime factor
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
        # Early termination: if best_val is 2, we can't do better for positive integers > 1
        # P(1)=1, but a+b=N odd, so one is even. If even one is 2, P=2.
        # If we find 2, it's likely minimal unless one is 1 (P=1) and other is 2 (P=2) -> max=2.
        # If a=1, b=N-1. P(1)=1. P(N-1) is large.
        # So 2 is the theoretical minimum if N-1 is a power of 2 (impossible here) or similar.
        # Actually, if a=2, b=N-2. P(2)=2.
        # If we find 2, we can break? No, 1 is possible?
        # P(a)=1 implies a=1. P(b)=1 implies b=1.
        # If a=1, b=N-1. max(1, P(N-1)).
        # If N-1 is 1, N=2. Here N=123457.
        # So min possible is likely >= 2.
        if best_val == 2:
            break
    
    b = N - best_a
    return [best_a, b]
