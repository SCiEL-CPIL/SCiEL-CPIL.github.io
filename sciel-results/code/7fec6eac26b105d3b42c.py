import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer for 120s limit
    
    # Generate powerful numbers up to L
    L = 10**7
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls = sorted(powerfuls)
    P = len(powerfuls)
    
    # Count sums
    sum_count = defaultdict(int)
    for i in range(P):
        a = powerfuls[i]
        if a > L:
            break
        for j in range(i, P):
            b = powerfuls[j]
            s = a + b
            if s > L * 2:
                break
            sum_count[s] += 1
    
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
# EVOLVE-BLOCK-END
