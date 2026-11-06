import math
from collections import Counter

def construct():
    # Limit for powerful numbers
    # We want to find N with many representations a+b=N
    # where a, b are powerful.
    # Generate powerful numbers up to L.
    # L = 10^7 allows ~6000 powerful numbers.
    # Pairs ~ 18 million.
    # This might take a few seconds.
    
    L = 10**7
    powerful_set = set()
    
    # Generate powerful numbers x = u^2 * v^3 <= L
    max_v = int(L ** (1/3))
    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > L:
            break
        max_u = int(math.isqrt(L // v3))
        for u in range(1, max_u + 1):
            x = u * u * v3
            if x <= L:
                powerful_set.add(x)
            else:
                break
    
    powerful_list = sorted(powerful_set)
    k = len(powerful_list)
    
    # Count sums
    # To speed up, we can break early if a+b > max_N?
    # We don't know max_N yet, but it's <= 2*L.
    # We just count all sums.
    
    sum_counts = Counter()
    
    # Optimization: use local variables
    pl = powerful_list
    n_pl = len(pl)
    
    for i in range(n_pl):
        a = pl[i]
        # b >= a
        for j in range(i, n_pl):
            b = pl[j]
            s = a + b
            # We can stop if s > some limit?
            # No, we want the global max.
            # But if s is very large, it's less likely to be the max?
            # Actually, smaller Ns have higher density?
            # No, density of powerful numbers decreases.
            # But the number of pairs for a specific N is what matters.
            sum_counts[s] += 1
    
    # Find N with max count
    # Tie-breaker: smaller N? Or larger N?
    # The task doesn't specify preference.
    # But smaller N might be safer?
    # Let's pick the one with max count.
    # If multiple, pick smallest N?
    
    best_N = None
    best_count = 0
    for s, c in sum_counts.items():
        if c > best_count:
            best_count = c
            best_N = s
        elif c == best_count and (best_N is None or s < best_N):
            best_N = s
            
    # Retrieve pairs for best_N
    pairs = []
    for i in range(n_pl):
        a = pl[i]
        b = best_N - a
        if b < a:
            continue
        if b in powerful_set:
            # Ensure b is in the list (it is, if b <= L)
            # But what if b > L?
            # Our generation was up to L.
            # If best_N > L, some b might be > L.
            # We should check if b is powerful.
            # We don't have a fast is_powerful function for large b.
            # So we should restrict N such that a, b <= L.
            # i.e., N <= 2*L.
            # And we only counted sums of numbers <= L.
            # So all pairs found are valid.
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
