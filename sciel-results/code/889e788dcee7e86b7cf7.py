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

    # Generate powerful numbers up to L
    # Every powerful number n = prod p_i^{e_i}, e_i >= 2
    # Can be written as a^2 * b^3 (not uniquely, but generates all)
    L = 8 * 10**7  # Target limit, tuned for time budget

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

    # If generation took too long, we still proceed with what we have
    # But if we have very few, try a smaller L
    if n_p < 100 and elapsed() < deadline - 80:
        L = 2 * 10**7
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
    # If n_p <= threshold, do full O(n^2) enumeration to find the mode of sumset
    # Otherwise, use sampling + two-pointer verification

    if n_p <= 12000:
        # Full O(n^2) enumeration
        count = defaultdict(int)
        for i in range(n_p):
            a = powerful[i]
            for j in range(i, n_p):
                b = powerful[j]
                N = a + b
                if N > 10**18:
                    break
                count[N] += 1
            if elapsed() > deadline - 25.0:
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
    # This is much better than uniform sampling because it targets actual sum values
    candidates = set()
    
    # Phase 1: Sample random pairs to get candidate N values
    n_samples = min(50000, n_p * 10)
    rng = random.Random(42)
    for _ in range(n_samples):
        if elapsed() > deadline - 60.0:
            break
        i = rng.randint(0, n_p - 1)
        j = rng.randint(i, n_p - 1)
        N = powerful[i] + powerful[j]
        if N <= 10**18:
            candidates.add(N)
    
    # Also add some structured candidates: sums of small powerful numbers with large ones
    # and vice versa, to cover different density regions
    for _ in range(min(5000, n_p)):
        if elapsed() > deadline - 55.0:
            break
        # Small a, large b
        i = rng.randint(0, min(100, n_p - 1))
        j = rng.randint(n_p - min(100, n_p), n_p - 1)
        if i <= j:
            N = powerful[i] + powerful[j]
            if N <= 10**18:
                candidates.add(N)
        # Large a, small b (same as above but reversed)
        i = rng.randint(n_p - min(100, n_p), n_p - 1)
        j = rng.randint(0, min(100, n_p - 1))
        if j <= i:
            N = powerful[i] + powerful[j]
            if N <= 10**18:
                candidates.add(N)
    
    # Also add candidates from consecutive powerful numbers (local structure)
    for i in range(min(5000, n_p - 1)):
        if elapsed() > deadline - 50.0:
            break
        N = powerful[i] + powerful[i + 1]
        if N <= 10**18:
            candidates.add(N)
    
    # Evaluate all candidates
    best_count = 0
    best_N = 0
    best_pairs = []
    
    candidate_list = list(candidates)
    # Sort candidates to process them in a reasonable order
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
    
    # Third pass: try to find even better candidates by sampling around the best region
    if best_count > 5 and elapsed() < deadline - 20.0:
        # Sample more candidates in the neighborhood of best_N
        extra_candidates = set()
        center = best_N
        spread = max(1000, best_count * 100)
        for _ in range(min(20000, n_p * 5)):
            if elapsed() > deadline - 10.0:
                break
            i = rng.randint(0, n_p - 1)
            j = rng.randint(i, n_p - 1)
            N = powerful[i] + powerful[j]
            if abs(N - center) <= spread and N <= 10**18:
                extra_candidates.add(N)
        
        for N in extra_candidates:
            if elapsed() > deadline - 5.0:
                break
            if N < 2 or N > 10**18:
                continue
            cnt, pairs = count_pairs_for_N(N)
            if cnt > best_count:
                best_count = cnt
                best_N = N
                best_pairs = pairs
        
        # Final local search
        if best_count > 0:
            fine_range3 = max(100, best_count * 30)
            for offset in range(-fine_range3, fine_range3 + 1):
                if elapsed() > deadline - 2.0:
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
        # Fallback: find at least one valid pair
        # Try N = 2 (1+1)
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
