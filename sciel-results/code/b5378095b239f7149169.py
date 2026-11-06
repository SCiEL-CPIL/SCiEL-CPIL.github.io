"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s buffer for 120s limit
    
    # Generate powerful numbers up to L
    L = 10**8
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
    n_pows = len(powerfuls)
    
    # Count sums of pairs (a, b) with a <= b
    sum_count = defaultdict(int)
    for i in range(n_pows):
        a = powerfuls[i]
        if time.time() > deadline - 3:
            break
        for j in range(i, n_pows):
            b_val = powerfuls[j]
            s = a + b_val
            if s > 2 * L:
                break
            sum_count[s] += 1
    
    # Find best N
    if not sum_count:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(sum_count, key=sum_count.get)
    
    # Collect pairs for best_N
    pows_set = set(powerfuls)
    pairs = []
    for a in powerfuls:
        if 2 * a > best_N:
            break
        b_val = best_N - a
        if b_val in pows_set and b_val >= a:
            pairs.append([a, b_val])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
