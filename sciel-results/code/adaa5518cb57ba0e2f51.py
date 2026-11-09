"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Generate powerful numbers up to L using a^2 * b^3 representation
    # Every powerful number can be written as a^2 * b^3 where b is cube-free
    # but we just enumerate all a^2 * b^3 <= L
    L = 2 * 10**7  # Keep moderate to ensure fast generation
    
    powerful_set = set()
    powerful_set.add(1)

    max_b = int(round(L ** (1.0/3.0))) + 2
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
    n_p = len(powerful)
    pset = set(powerful)
    
    # Structural change: Use hash-based pair counting with strategic N sampling
    # Instead of FFT, we sample candidate N values and count pairs in O(n) each
    # This allows us to explore a much wider range of N values
    
    best_N = 2
    best_count = 0
    best_pairs = [[1, 1]]
    
    # Phase 1: Sample N values from the sumset
    # The sum of two powerful numbers ranges from 2 to 2*L
    # We'll sample N values at various scales to find dense regions
    
    # Generate candidate N values by sampling sums of random pairs
    candidates = set()
    num_samples = min(50000, n_p * 20)
    for _ in range(num_samples):
        if elapsed() > deadline - 60:
            break
        i = random.randint(0, n_p - 1)
        j = random.randint(0, n_p - 1)
        N = powerful[i] + powerful[j]
        if 2 <= N <= 10**18:
            candidates.add(N)
    
    # Also add some structured candidates: multiples of small powerful numbers
    for small_p in powerful[:20]:
        for mult in range(1, min(100, (2*L) // max(small_p, 1))):
            N = small_p * mult
            if 2 <= N <= 2 * L:
                candidates.add(N)
    
    # Count pairs for each candidate N using two-pointer
    for N in candidates:
        if elapsed() > deadline - 30:
            break
        if N < 2 or N > 10**18:
            continue
        
        # Two-pointer count
        i = 0
        j = n_p - 1
        cnt = 0
        while i <= j:
            s = powerful[i] + powerful[j]
            if s == N:
                cnt += 1
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        
        if cnt > best_count:
            best_count = cnt
            best_N = N
            # Extract pairs
            pairs = []
            i = 0
            j = n_p - 1
            while i <= j:
                s = powerful[i] + powerful[j]
                if s == N:
                    pairs.append([powerful[i], powerful[j]])
                    i += 1
                    j -= 1
                elif s < N:
                    i += 1
                else:
                    j -= 1
            best_pairs = pairs
    
    # Phase 2: Local search around best_N
    if best_count > 0 and elapsed() < deadline - 20:
        search_radius = max(50, best_count * 10)
        for offset in range(1, search_radius + 1):
            if elapsed() > deadline - 10:
                break
            for N in [best_N + offset, best_N - offset]:
                if N < 2 or N > 10**18:
                    continue
                
                i = 0
                j = n_p - 1
                cnt = 0
                while i <= j:
                    s = powerful[i] + powerful[j]
                    if s == N:
                        cnt += 1
                        i += 1
                        j -= 1
                    elif s < N:
                        i += 1
                    else:
                        j -= 1
                
                if cnt > best_count:
                    best_count = cnt
                    best_N = N
                    pairs = []
                    i = 0
                    j = n_p - 1
                    while i <= j:
                        s = powerful[i] + powerful[j]
                        if s == N:
                            pairs.append([powerful[i], powerful[j]])
                            i += 1
                            j -= 1
                        elif s < N:
                            i += 1
                        else:
                            j -= 1
                    best_pairs = pairs
    
    # Phase 3: If count is still low, try O(n^2) for smaller n_p
    if best_count < 5 and n_p <= 5000 and elapsed() < deadline - 30:
        count = defaultdict(int)
        for i in range(n_p):
            a = powerful[i]
            for j in range(i, n_p):
                b = powerful[j]
                N = a + b
                if N > 10**18:
                    break
                count[N] += 1
            if elapsed() > deadline - 15:
                break
        
        if count:
            best_N = max(count, key=count.get)
            best_count = count[best_N]
            best_pairs = []
            for a in powerful:
                b = best_N - a
                if b >= a and b in pset:
                    best_pairs.append([a, b])
    
    if not best_pairs:
        best_pairs = [[1, 1]]
        best_N = 2
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
