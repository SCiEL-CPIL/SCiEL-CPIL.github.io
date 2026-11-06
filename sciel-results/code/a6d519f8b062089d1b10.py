def construct():
    from math import isqrt

    # Increase L to generate more powerful numbers
    # L = 2 * 10^7 implies M ~ 60,000
    # M^2 / 2 ~ 1.8 * 10^9 operations
    # Using list for counts is much faster than dict
    L = 20 * 10**6
    
    # Generate square-free numbers up to L^(1/3)
    max_b = int(L ** (1/3)) + 1
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
    
    # Use a list for counts to speed up indexing
    # Max sum is 2 * L
    max_sum = 2 * L
    counts = [0] * (max_sum + 1)
    
    # Count sums
    P_local = P
    for i in range(M):
        a = P_local[i]
        for j in range(i, M):
            s = a + P_local[j]
            counts[s] += 1
            
    # Find best N
    best_count = max(counts)
    best_N = counts.index(best_count)
    
    # Collect pairs
    pairs = []
    ps_set = powerful_set
    for i in range(M):
        a = P_local[i]
        b = best_N - a
        if b < a: continue
        if b in ps_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
