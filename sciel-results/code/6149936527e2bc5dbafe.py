"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import isqrt
from collections import defaultdict

def _generate_powerfuls(limit):
    """Generate all powerful numbers up to limit."""
    pows = set()
    # Every powerful number can be written as a^2 * b^3 where b is squarefree
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                pows.add(n)
    return pows

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Generate powerful numbers up to a large limit
    L = 50_000_000  # 5 * 10^7
    powerfuls = _generate_powerfuls(L)
    powerfuls_list = sorted(powerfuls)
    n_pows = len(powerfuls_list)
    pows_set = set(powerfuls_list)
    
    # Sparse sum counting: for each pair (a, b) with a <= b, count sum
    # Use a dictionary to count sums
    sum_counts = defaultdict(int)
    
    # Iterate over pairs efficiently
    # For each a in powerfuls_list, for each b >= a in powerfuls_list with a+b <= 2*L
    # But this is O(n^2) which is too slow for large n
    
    # Better approach: for each a, binary search for valid b range
    # Then count how many b's give each sum... but sums are unique per (a,b) pair
    
    # Actually, let's use the fact that we want to find N with many representations.
    # Strategy: sample candidate N values and count representations.
    
    # First, let's try a hybrid: use FFT if feasible, else sparse search
    # For L=5e7, FFT size would be 2^27 = 134M floats = ~1GB. Too much.
    
    # Sparse approach: iterate over powerful numbers and count pair sums
    # For n_pows ~ 50k-100k, O(n^2) is too slow.
    
    # Better: for each powerful number a, the possible b values are powerful numbers
    # in [a, 2*L - a]. We can enumerate these and count sums.
    
    # Even better: use a two-pointer or sorted approach
    # For a fixed N, count = number of a in powerfuls such that N-a is also in powerfuls and a <= N-a
    
    # Let's find good candidate N values by sampling sums of random pairs
    # Then verify the top candidates
    
    # Phase 1: Generate candidate N values from random pair sums
    import random
    random.seed(42)
    
    candidates = defaultdict(int)
    # Sample many random pairs to find promising N values
    n_samples = min(500000, n_pows * 10)
    for _ in range(n_samples):
        i = random.randint(0, n_pows - 1)
        j = random.randint(0, n_pows - 1)
        a = powerfuls_list[i]
        b = powerfuls_list[j]
        if a > b:
            a, b = b, a
        s = a + b
        if s <= 2 * L:
            candidates[s] += 1
    
    # Phase 2: Get top candidate N values
    if not candidates:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Sort by count, take top candidates
    top_candidates = sorted(candidates.items(), key=lambda x: -x[1])[:50]
    
    # Phase 3: Verify and count exactly for top candidates
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    for N_cand, _ in top_candidates:
        if time.time() > deadline:
            break
        
        N_val = N_cand
        # Count exact representations
        pairs = []
        for a in powerfuls_list:
            if 2 * a > N_val:
                break
            b_val = N_val - a
            if b_val in pows_set and b_val >= a:
                pairs.append([a, b_val])
        
        if len(pairs) > best_count:
            best_count = len(pairs)
            best_N = N_val
            best_pairs = pairs
    
    # Phase 4: If we have time, do a more systematic search around best_N
    if time.time() < deadline - 5 and best_count > 0:
        # Search in a window around best_N
        search_radius = 1000
        for offset in range(-search_radius, search_radius + 1):
            if time.time() > deadline:
                break
            N_val = best_N + offset
            if N_val < 2 or N_val > 2 * L:
                continue
            # Quick count
            cnt = 0
            for a in powerfuls_list:
                if 2 * a > N_val:
                    break
                b_val = N_val - a
                if b_val in pows_set and b_val >= a:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_N = N_val
                # Collect pairs
                best_pairs = []
                for a in powerfuls_list:
                    if 2 * a > N_val:
                        break
                    b_val = N_val - a
                    if b_val in pows_set and b_val >= a:
                        best_pairs.append([a, b_val])
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
