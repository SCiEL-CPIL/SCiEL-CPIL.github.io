"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    def elapsed_ok():
        return time.time() < deadline
    
    def generate_powerful(limit):
        result = set()
        max_b = int(limit ** (1/3)) + 1
        for b in range(1, max_b):
            b3 = b * b * b
            if b3 > limit:
                break
            max_a = isqrt(limit // b3)
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val > limit:
                    break
                result.add(val)
        return sorted(result)
    
    # Generate powerful numbers up to 10^8
    gen_limit = 10**8
    powerful = generate_powerful(gen_limit)
    powerful_set = set(powerful)
    
    # Count sum frequencies
    sum_count = {}
    K = len(powerful)
    
    for i in range(K):
        if not elapsed_ok():
            break
        pi = powerful[i]
        for j in range(i, K):
            N = pi + powerful[j]
            sum_count[N] = sum_count.get(N, 0) + 1
    
    # Find N with max count
    best_N = 2
    best_count = 0
    for N, cnt in sum_count.items():
        if cnt > best_count:
            best_count = cnt
            best_N = N
    
    # Extract pairs for best_N
    pairs = []
    for p in powerful:
        if p > best_N // 2:
            break
        q = best_N - p
        if q in powerful_set:
            pairs.append([p, q])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
