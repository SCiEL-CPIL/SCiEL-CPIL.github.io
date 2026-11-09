import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # 1. Generate powerful numbers up to M
    # We choose M to balance search space and computation time.
    # M = 10^8 gives ~26k numbers, ~3.4e8 pairs. Should run in ~30-40s in PyPy/CPython.
    M = 100000000  # 10^8
    
    # Precompute square-free numbers up to M^(1/3)
    max_v = int(M ** (1/3)) + 2
    is_sq_free = [True] * (max_v + 1)
    is_sq_free[0] = False
    if max_v >= 1:
        is_sq_free[1] = True
    for p in range(2, int(math.isqrt(max_v)) + 1):
        if is_sq_free[p]:
            p2 = p * p
            for j in range(p2, max_v + 1, p2):
                is_sq_free[j] = False
    
    square_free_v = [v for v in range(1, max_v + 1) if is_sq_free[v]]
    
    powerful_set = set()
    powerful_list = []
    
    for v in square_free_v:
        v3 = v ** 3
        if v3 > M:
            continue
        max_u = int(math.isqrt(M // v3))
        for u in range(1, max_u + 1):
            x = u * u * v3
            if x <= M:
                if x not in powerful_set:
                    powerful_set.add(x)
                    powerful_list.append(x)
    
    powerful_list.sort()
    L = len(powerful_list)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 2. Count pair sums
    counts = {}
    best_N = 0
    best_count = 0
    
    P_list = powerful_list
    # Optimization: Use local variables for speed
    for i in range(L):
        ai = P_list[i]
        # j must be >= i to ensure a <= b
        for j in range(i, L):
            s = ai + P_list[j]
            c = counts.get(s, 0) + 1
            counts[s] = c
            if c > best_count:
                best_count = c
                best_N = s
                
        # Time check every 1000 iterations of outer loop
        if i % 1000 == 0 and time.time() > deadline:
            break

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 3. Collect pairs for best_N
    pairs = []
    P_set = powerful_set
    for a in P_list:
        b = best_N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([a, b])
            
    # Ensure distinct pairs and a <= b
    # The loop above ensures a <= b. 
    # Since P_list has unique elements, pairs are unique.
    
    return {'N': best_N, 'pairs': pairs}
