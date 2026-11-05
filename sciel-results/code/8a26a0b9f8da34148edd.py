import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We want to finish by 110s.
    # Generation takes ~1s. Counting takes the rest.
    timeout = 105
    
    # M = 200,000,000 allows for K ~ 15,000 powerful numbers.
    # Pairs ~ 112,500,000.
    # This should take ~20-30 seconds in Python.
    M = 200_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    # Generate powerful numbers
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
    n_list = len(powerful_list)
    
    # Count sums
    counts = {}
    pl = powerful_list
    
    # Use a local variable for time check to reduce overhead
    # Check time every 1000 iterations of the outer loop
    for i in range(n_list):
        a = pl[i]
        if i % 1000 == 0 and time.time() - start_time > timeout:
            break
            
        for j in range(i, n_list):
            s = a + pl[j]
            counts[s] = counts.get(s, 0) + 1
            
    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
