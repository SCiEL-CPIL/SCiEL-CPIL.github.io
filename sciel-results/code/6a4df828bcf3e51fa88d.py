def construct():
    N = 123457
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute largest prime factor P(n) for all n
    P = [0] * (N + 1)
    P[1] = 1
    for i in range(2, N + 1):
        p = spf[i]
        P[i] = max(p, P[i // p])
        
    # Find the pair minimizing max(P(a), P(b))
    best_score = float('inf')
    best_a = 1
    # We can stop at N//2 due to symmetry, but full loop is fine
    for a in range(1, N // 2 + 1):
        b = N - a
        score = max(P[a], P[b])
        if score < best_score:
            best_score = score
            best_a = a
            
    return [best_a, N - best_a]
