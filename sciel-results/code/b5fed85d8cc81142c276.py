import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Generate powerful numbers up to L
    L = 10**12
    
    powerful_set = set()
    powerful_set.add(1)
    
    # n = a^2 * b^3
    max_b = int(round(L ** (1/3))) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerful_set.add(n)
    
    powerful = sorted(powerful_set)
    n_powerful = len(powerful)
    
    if time.time() > deadline - 60.0:
        # Fallback to smaller L if generation took too long
        L = 10**9
        powerful_set = {1}
        max_b = int(round(L ** (1/3))) + 1
        for b in range(1, max_b + 1):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful_set.add(n)
        powerful = sorted(powerful_set)

    # Search for best N in [L-100, L]
    # Theoretical optimum is around L
    best_N = L
    best_count = 0
    best_pairs = []
    
    # We check a window around L
    # Since r(N) is roughly constant or slightly decreasing for N > L/2,
    # we prioritize larger N.
    # However, specific N might have more due to structure.
    
    search_range = range(L, L - 200, -1)
    
    for N in search_range:
        if time.time() > deadline - 30.0:
            break
            
        count = 0
        pairs = []
        # a <= b => a <= N/2
        # a must be in powerful_set
        # N-a must be in powerful_set
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    if not best_pairs:
        # Fallback if no pairs found (should not happen)
        best_N = 2
        best_pairs = [[1, 1]]
        
    return {'N': best_N, 'pairs': best_pairs}
