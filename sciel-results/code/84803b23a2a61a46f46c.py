import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    # Limit L
    L = 5 * 10**7
    
    powerfuls = set()
    
    # Generate square-free numbers up to L^(1/3)
    # v^3 <= L => v <= L^(1/3)
    max_v = int(L ** (1/3)) + 2
    
    # Sieve for square-free
    is_square_free = [True] * (max_v + 1)
    is_square_free[0] = False
    is_square_free[1] = True # 1 is square-free
    
    for p in range(2, isqrt(max_v) + 1):
        if is_square_free[p]:
            p2 = p * p
            for k in range(p2, max_v + 1, p2):
                is_square_free[k] = False
                
    # Generate powerful numbers
    for v in range(1, max_v + 1):
        if not is_square_free[v]:
            continue
        v3 = v ** 3
        if v3 > L:
            break
        max_u = isqrt(L // v3)
        for u in range(1, max_u + 1):
            n = u * u * v3
            if n <= L:
                powerfuls.add(n)
                
    powerfuls = sorted(powerfuls)
    P = len(powerfuls)
    
    # Count sums
    sum_count = defaultdict(int)
    for i in range(P):
        a = powerfuls[i]
        # Optimization: if a is large, b must be small?
        # No, we just iterate all j >= i
        for j in range(i, P):
            b = powerfuls[j]
            s = a + b
            # We don't strictly need to bound s, but N <= 2L
            # If we want to be faster, we can break if s > some limit?
            # But we want the global max.
            sum_count[s] += 1
            
            if time.time() > deadline - 5:
                # Emergency exit
                break
        if time.time() > deadline - 5:
            break

    # Find best N
    best_N = 2
    best_count = 0
    for n, cnt in sum_count.items():
        if cnt > best_count:
            best_count = cnt
            best_N = n

    # Generate pairs for best_N
    members = set(powerfuls)
    pairs = []
    for a in powerfuls:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in members and a <= b:
            pairs.append([a, b])
    
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
