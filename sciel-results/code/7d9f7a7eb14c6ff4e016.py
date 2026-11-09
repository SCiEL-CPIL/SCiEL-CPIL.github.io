"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Generate powerful numbers up to L
    # Every powerful number n = prod p_i^{e_i}, e_i >= 2
    # Can be written as a^2 * b^3 (not uniquely, but generates all)
    # We want L large enough to have a dense set of powerful numbers.
    # The count of powerful numbers up to L is roughly O(L^(1/3)) or O(L^(1/2))?
    # Actually, it is known that the number of powerful numbers <= x is ~ x^(1/3) * constant.
    # Let's test with L = 10^9.
    # If count is ~ 1000-10000, O(n^2) is feasible.
    
    L = 10**9

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
    
    # If generation took too long or n_p is too small, try smaller L
    if elapsed() > 20 or n_p < 500:
        L = 10**8
        powerful_set = set([1])
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

    # Strategy:
    # 1. If n_p is small enough (e.g., < 5000), do full O(n^2) counting using NumPy for speed.
    # 2. If n_p is larger, use sampling + two-pointer verification.
    
    if n_p <= 5000:
        # Full O(n^2) enumeration using NumPy
        # We can't easily do O(n^2) in one shot if n_p=5000 (25M ops) might be slow in Python loop.
        # But NumPy vectorization helps if we do it in chunks or use np.add.outer?
        # np.add.outer would create a 5000x5000 array = 25M elements = 200MB. Feasible.
        # But we need to count sums. 
        # Better: iterate i, and for each i, compute powerful[i:] + powerful[i] and bincount?
        # Or just use a defaultdict in Python if n_p is small?
        # 5000^2 / 2 = 12.5M iterations. In Python, this might take 1-2 seconds.
        # Let's try the Python loop first, it's simpler and memory safe.
        
        count = defaultdict(int)
        P = powerful
        for i in range(n_p):
            a = P[i]
            for j in range(i, n_p):
                b = P[j]
                N = a + b
                if N > 10**18:
                    break
                count[N] += 1
            if elapsed() > deadline - 10.0:
                break

        if count:
            best_N = max(count, key=count.get)
            best_count = count[best_N]
            pset = set(powerful)
            pairs = []
            for a in powerful:
                b = best_N - a
                if b >= a and b in pset:
                    pairs.append([a, b])
            return {'N': best_N, 'pairs': pairs}

    # For larger n_p: use two-pointer counting with smart candidate generation
    def count_pairs_for_N(N):
        """Count pairs (a,b) with a<=b, a+b=N, both powerful, using two pointers."""
        i = 0
        j = n_p - 1
        count = 0
        pairs = []
        while i <= j:
            s = powerful[i] + powerful[j]
            if s == N:
                count += 1
                pairs.append([powerful[i], powerful[j]])
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        return count, pairs

    # Generate candidate N values by sampling pairs
    candidates = set()
    
    import random
    rng = random.Random(42)
    
    # Phase 1: Sample random pairs to get candidate N values
    n_samples = min(200000, n_p * 10)
    for _ in range(n_samples):
        if elapsed() > deadline - 60.0:
            break
        i = rng.randint(0, n_p - 1)
        j = rng.randint(i, n_p - 1)
        N = powerful[i] + powerful[j]
        if N <= 10**18:
            candidates.add(N)
    
    # Also add some structured candidates: sums of small powerful numbers with large ones
    for _ in range(min(10000, n_p)):
        if elapsed() > deadline - 55.0:
            break
        # Small a, large b
        i = rng.randint(0, min(100, n_p - 1))
        j = rng.randint(n_p - min(100, n_p), n_p - 1)
        if i <= j:
            N = powerful[i] + powerful[j]
            if N <= 10**18:
                candidates.add(N)
        
    # Evaluate all candidates
    best_count = 0
    best_N = 0
    best_pairs = []
    
    candidate_list = list(candidates)
    candidate_list.sort()
    
    for N in candidate_list:
        if elapsed() > deadline - 30.0:
            break
        if N < 2 or N > 10**18:
            continue
        cnt, pairs = count_pairs_for_N(N)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    
    # Fine search around the best N found
    if best_count > 0:
        fine_range = max(200, best_count * 50)
        for offset in range(-fine_range, fine_range + 1):
            if elapsed() > deadline - 10.0:
                break
            N = best_N + offset
            if N < 2 or N > 10**18:
                continue
            cnt, pairs = count_pairs_for_N(N)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs
    
    # Second refinement pass with wider range if we found something good
    if best_count > 3 and elapsed() < deadline - 15.0:
        fine_range2 = max(500, best_count * 80)
        for offset in range(-fine_range2, fine_range2 + 1):
            if elapsed() > deadline - 5.0:
                break
            N = best_N + offset
            if N < 2 or N > 10**18:
                continue
            cnt, pairs = count_pairs_for_N(N)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs
    
    if not best_pairs:
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
