def construct():
    from math import isqrt
    import collections

    # Set limit for powerful numbers generation
    # Target M ~ 20,000 powerful numbers
    # M ~ C * L^(2/3) -> L ~ (M/C)^(3/2)
    # C is approx 1.
    # Let's try L = 10^7 first.
    L = 10**7
    
    # Generate square-free numbers up to L^(1/3)
    max_b = int(L ** (1/3)) + 1
    square_free = []
    # Simple sieve for square-free
    is_sq_free = [True] * (max_b + 1)
    is_sq_free[0] = False
    is_sq_free[1] = True
    for i in range(2, isqrt(max_b) + 1):
        if is_sq_free[i]:
            sq = i * i
            for j in range(sq, max_b + 1, sq):
                is_sq_free[j] = False
    square_free = [i for i in range(1, max_b + 1) if is_sq_free[i]]
    
    powerful_set = set()
    for b in square_free:
        b3 = b ** 3
        if b3 > L: break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            powerful_set.add(a * a * b3)
            
    P = sorted(powerful_set)
    M = len(P)
    
    # Find best N
    counts = collections.defaultdict(int)
    # To speed up, we can break early if a > N/2? No, we don't know N.
    # Just iterate all pairs.
    # Optimization: use list indexing
    for i in range(M):
        a = P[i]
        # b >= a
        for j in range(i, M):
            b = P[j]
            s = a + b
            counts[s] += 1
            
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Collect pairs
    pairs = []
    for i in range(M):
        a = P[i]
        b = best_N - a
        if b < a: continue
        if b in powerful_set:
            # Ensure b is in P? If b is powerful and b <= L, it is in P.
            # But what if b > L?
            # If a <= L and b = N-a.
            # If N is sum of two powerful numbers <= L, then b <= L.
            # Our counts only include pairs from P, so b must be in P.
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
