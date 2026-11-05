import time
from math import isqrt
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    def time_left():
        return deadline - time.time()
    
    # Generate powerful numbers up to limit M
    # Powerful: n = a^2 * b^3 where b is square-free
    # Use sieve-based square-free detection for efficiency
    
    M = 100_000_000  # 100 million - balance between count and speed
    
    max_b = int(M ** (1.0 / 3.0)) + 2  # ~465
    
    # Precompute square-free status for all numbers up to max_b using sieve
    sq_free = [True] * (max_b + 1)
    sq_free[0] = False
    sq_free[1] = True  # 1 is square-free
    for p in range(2, isqrt(max_b) + 1):
        p2 = p * p
        if sq_free[p]:  # p is prime (not yet marked by smaller prime square)
            for j in range(p2, max_b + 1, p2):
                sq_free[j] = False
    
    powerful = set()
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b_cubed = b ** 3
        if b_cubed > M:
            break
        max_a = isqrt(M // b_cubed)
        # Generate all a^2 * b_cubed <= M
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
    
    if time.time() > deadline - 40:
        # Fallback to smaller M
        M = 30_000_000
        max_b = int(M ** (1.0 / 3.0)) + 2
        sq_free = [True] * (max_b + 1)
        sq_free[0] = False
        sq_free[1] = True
        for p in range(2, isqrt(max_b) + 1):
            p2 = p * p
            if sq_free[p]:
                for j in range(p2, max_b + 1, p2):
                    sq_free[j] = False
        powerful = set()
        for b in range(1, max_b + 1):
            if not sq_free[b]:
                continue
            b_cubed = b ** 3
            if b_cubed > M:
                break
            max_a = isqrt(M // b_cubed)
            for a in range(1, max_a + 1):
                powerful.add(a * a * b_cubed)
    
    powerful_list = sorted(powerful)
    powerful_set = powerful
    K = len(powerful_list)
    
    if time.time() > deadline - 15:
        # Emergency: use small subset
        subset = powerful_list[:1000]
        pw_set = set(subset)
        counts = defaultdict(int)
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                N = subset[i] + subset[j]
                counts[N] += 1
        best_N = max(counts, key=counts.get) if counts else 2
        pairs = []
        for a in subset:
            if 2 * a > best_N:
                break
            b = best_N - a
            if b in pw_set and a <= b:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    
    # Efficient pair counting: O(K^2) with Python dict
    # K is ~2500-4000 for M=100M, so K^2/2 ~ 3-8M ops, very fast
    counts = defaultdict(int)
    best_N = 0
    best_count = 0
    
    # Use local variables for speed
    pw_list = powerful_list
    pw_set = powerful_set
    
    for i in range(K):
        a = pw_list[i]
        for j in range(i, K):
            N = a + pw_list[j]
            c = counts[N] + 1
            counts[N] = c
            if c > best_count:
                best_count = c
                best_N = N
        
        if time.time() > deadline - 20:
            break
    
    if best_count == 0:
        best_N = 2
        best_count = 1
    
    # Rebuild pairs for best_N
    pairs = []
    for a in powerful_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
