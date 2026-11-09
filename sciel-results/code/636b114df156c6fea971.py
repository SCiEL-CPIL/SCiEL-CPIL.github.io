def construct():
    from math import isqrt
    L = 10**9
    powerful = set()
    # v^3 <= L
    max_v = isqrt(isqrt(L)) # Wait, cube root
    # 10^9 ** (1/3) = 1000
    max_v = 1000
    for v in range(1, max_v + 1):
        v3 = v**3
        # u^2 <= L / v3
        max_u = isqrt(L // v3)
        for u in range(1, max_u + 1):
            powerful.add(u*u * v3)
    
    P = sorted(powerful)
    n = len(P)
    
    counts = {}
    # Optimization: local variables
    P_local = P
    counts_local = counts
    for i in range(n):
        pi = P_local[i]
        for j in range(i, n):
            s = pi + P_local[j]
            if s in counts_local:
                counts_local[s] += 1
            else:
                counts_local[s] = 1
                
    # Find max
    best_N = 0
    best_count = 0
    for k, v in counts.items():
        if v > best_count:
            best_count = v
            best_N = k
            
    # Find pairs
    pairs = []
    members = set(P)
    for a in P:
        b = best_N - a
        if b < a:
            break
        if b in members:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
