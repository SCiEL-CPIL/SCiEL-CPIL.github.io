"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, gcd
from collections import defaultdict

def _gen_powerful(limit):
    """Generate all powerful numbers <= limit."""
    pows = set()
    b_max = int(limit ** (1/3)) + 1
    
    is_sqfree = [True] * (b_max + 1)
    if b_max >= 0:
        is_sqfree[0] = False
    if b_max >= 1:
        is_sqfree[1] = True
    for i in range(2, isqrt(b_max) + 1):
        if is_sqfree[i]:
            sq = i * i
            for j in range(sq, b_max + 1, sq):
                is_sqfree[j] = False
    
    squarefree_b = [i for i in range(1, b_max + 1) if is_sqfree[i]]
    
    for b in squarefree_b:
        b3 = b ** 3
        if b3 > limit:
            continue
        a_max = isqrt(limit // b3)
        for a in range(1, a_max + 1):
            n = a * a * b3
            if n <= limit:
                pows.add(n)
    
    pows.add(1)
    return sorted(pows)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    L = 10**9
    powerful = _gen_powerful(L)
    pow_set = set(powerful)
    n_pow = len(powerful)
    
    import numpy as np
    pow_arr = np.array(powerful, dtype=np.int64)
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Strategy: Find N values that are sums of many small powerful numbers
    # Use the most frequent sums of small powerful numbers as candidates
    
    # Take the smallest powerful numbers (they'll be in many sums)
    small_pow = powerful[:2000]
    
    # Compute all pairwise sums of small powerful numbers
    # This gives us candidate N values that have many representations
    sum_counts = defaultdict(int)
    sum_pairs = defaultdict(list)
    
    for i, a in enumerate(small_pow):
        for j in range(i, len(small_pow)):
            b = small_pow[j]
            s = a + b
            if s <= 2 * L:
                sum_counts[s] += 1
                if len(sum_pairs[s]) < 50:  # limit storage
                    sum_pairs[s].append((a, b))
    
    # Sort candidates by count
    sorted_candidates = sorted(sum_counts.items(), key=lambda x: -x[1])
    
    # Also add some structured candidates
    extra_candidates = []
    for p in powerful[:100]:
        extra_candidates.append(2 * p)
        extra_candidates.append(p + 1)
        if p + 2 <= 2 * L:
            extra_candidates.append(p + 2)
    
    # Combine and deduplicate
    all_candidates = list(set([c[0] for c in sorted_candidates[:500]] + extra_candidates))
    all_candidates.sort()
    
    # Evaluate each candidate
    for N in all_candidates:
        if time.time() > deadline - 30:
            break
        if N < 2 or N > 10**18:
            continue
        
        # Count pairs: a <= b, a + b = N, a,b in powerful
        a_lo = max(1, N - L)
        a_hi = min(L, N // 2)
        
        if a_lo > a_hi:
            continue
        
        left = np.searchsorted(pow_arr, a_lo, side='left')
        right = np.searchsorted(pow_arr, a_hi, side='right')
        
        if right <= left:
            continue
        
        a_candidates = pow_arr[left:right]
        b_candidates = N - a_candidates
        
        # Vectorized membership check using searchsorted
        insert_pos = np.searchsorted(pow_arr, b_candidates)
        valid_mask = (insert_pos < len(pow_arr)) & (pow_arr[np.minimum(insert_pos, len(pow_arr)-1)] == b_candidates)
        
        count = int(np.sum(valid_mask))
        
        if count > best_count:
            best_count = count
            best_N = N
            valid_a = a_candidates[valid_mask]
            best_pairs = [[int(a), int(N - a)] for a in valid_a]
    
    # If we found something, verify and return
    if best_N is not None and best_count > 0:
        return {'N': best_N, 'pairs': best_pairs}
    
    # Fallback
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
