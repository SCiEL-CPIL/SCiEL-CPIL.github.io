import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 100
    
    # Target M
    M = 20_000_000
    
    powerful = set()
    # y is square-free
    # p = x^2 * y^3
    # y^3 <= M => y <= M^(1/3)
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        # Check if y is square-free
        # We can just generate and filter, or check prime factors.
        # Simpler: just generate x^2 * y^3 and add to set. Duplicates are handled by set.
        # But we need to ensure y is square-free?
        # If y is not square-free, y^3 has a prime factor with power >= 3.
        # x^2 * y^3 is powerful regardless.
        # The set of powerful numbers is exactly {x^2 y^3 | x, y >= 1}.
        # So we don't need y to be square-free for generation, just for uniqueness of representation if we were counting reps.
        # We just need the set of values.
        
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            powerful.add(p)
            
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    
    # Count sums
    counts = {}
    n_list = len(powerful_list)
    
    for i in range(n_list):
        a = powerful_list[i]
        if time.time() - start_time > timeout - 10:
            break
        for j in range(i, n_list):
            b = powerful_list[j]
            s = a + b
            # s is always <= 2*M which is 4e7, well within 1e18
            counts[s] = counts.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
