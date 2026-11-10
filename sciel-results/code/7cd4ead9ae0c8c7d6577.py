import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    N = 123457
    
    # Build smallest prime factor (SPF) sieve up to N
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute P(n) = largest prime factor of n, for all n in [0, N]
    P = [0] * (N + 1)
    P[0] = 0
    P[1] = 1
    for n in range(2, N + 1):
        p = spf[n]
        rest = n // p
        P[n] = max(p, P[rest])
    
    # Search for optimal a in [1, N-1] minimizing max(P(a), P(N-a))
    best_val = float('inf')
    best_a = 1
    for a in range(1, N):
        b = N - a
        val = max(P[a], P[b])
        if val < best_val:
            best_val = val
            best_a = a
        if time.time() > deadline:
            break
    
    return [best_a, N - best_a]
