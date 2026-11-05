import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 110
    
    # Increase M to explore larger N for potentially higher representation counts
    M = 50_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            powerful.add(p)
            
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    
    counts = {}
    n_list = len(powerful_list)
    p_list = powerful_list
    
    for i in range(n_list):
        a = p_list[i]
        # Periodic time check to respect deadline
        if i % 1000 == 0 and time.time() - start_time > timeout - 20:
            break
        for j in range(i, n_list):
            s = a + p_list[j]
            counts[s] = counts.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    
    pairs = []
    p_set = powerful_set
    for a in p_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
