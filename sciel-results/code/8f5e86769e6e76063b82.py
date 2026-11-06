"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from collections import Counter
import numpy as np

def _gen_powerful(limit):
    """Generate all powerful numbers <= limit using a^2 * b^3 form."""
    pows = set()
    b_max = int(limit ** (1/3)) + 2
    
    # Sieve for squarefree numbers up to b_max
    is_sqfree = [True] * (b_max + 1)
    if b_max >= 0:
        is_sqfree[0] = False
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
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Phase 1: Generate powerful numbers up to a reasonable limit
    L = 10**9
    powerful = _gen_powerful(L)
    pow_set = set(powerful)
    n_pow = len(powerful)
    pow_arr = np.array(powerful, dtype=np.int64)
    
    # Phase 2: Frequency-based candidate identification
    # Compute all pairwise sums a+b where a <= b, a,b in powerful, a+b <= 2*L
    # Use numpy for efficiency: for each a in powerful, compute a + powerful[a_idx:]
    # But that's O(n^2) which is too large for n ~ 10^5
    
    # Better: use a smarter approach
    # For each powerful number a, the valid b values are those with b >= a and a+b <= 2*L
    # So b is in [a, 2*L - a]
    # We want to count how many powerful numbers are in [a, 2*L-a] for each a
    
    # Actually, let's compute frequencies of sums more efficiently
    # Use the fact that for a given sum S, the number of representations is
    # |{a in powerful : a <= S/2 and S-a in powerful}|
    
    # Strategy: compute all sums using convolution-like approach
    # Create a boolean array of size 2*L+1, mark powerful numbers
    # Then for each candidate S, count = sum over a of mark[a] * mark[S-a]
    # This is like a convolution
    
    # But 2*L = 2*10^9 is too large for a dense array
    # Instead, use the sorted list and binary search
    
    # Phase 3: Identify top candidates by sampling sums
    # Sample random pairs (a, b) with a <= b, a, b in powerful
    # Record the sum frequencies
    # The sums with highest frequency are our candidates
    
    rng = np.random.default_rng(42)
    n_samples = 500000
    idx_a = rng.integers(0, n_pow, size=n_samples)
    idx_b = rng.integers(0, n_pow, size=n_samples)
    
    # Ensure a <= b
    swap = idx_a > idx_b
    idx_a[swap], idx_b[swap] = idx_b[swap].copy(), idx_a[swap].copy()
    
    sums = pow_arr[idx_a] + pow_arr[idx_b]
    sums = sums[sums <= 2 * L]
    
    # Count frequencies
    if len(sums) > 0:
        sum_counts = Counter(sums.tolist())
        top_candidates = [s for s, c in sum_counts.most_common(200)]
    else:
        top_candidates = []
    
    # Also add some structured candidates
    candidates = set(top_candidates)
    for p in powerful[:200]:
        candidates.add(2 * p)
        candidates.add(p + 1)
        if p + 2 <= 2 * L:
            candidates.add(p + 2)
    
    # Add candidates near 2*L
    for offset in range(0, 500, 5):
        candidates.add(2 * L - offset)
        candidates.add(2 * L + offset)
    
    candidates = sorted([c for c in candidates if 2 <= c <= 10**18])
    
    # Phase 4: Evaluate candidates
    best_N = None
    best_count = 0
    best_pairs = []
    
    for N in candidates:
        if time.time() > deadline - 10:
            break
        if N < 2 or N > 10**18:
            continue
        
        # Count pairs: a <= b, a + b = N, a,b powerful
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
        
        # Check which b_candidates are in pow_set
        mask = np.array([b in pow_set for b in b_candidates])
        count = int(np.sum(mask))
        
        if count > best_count:
            best_count = count
            best_N = N
            valid_a = a_candidates[mask]
            best_pairs = [[int(a), int(N - a)] for a in valid_a]
    
    # If we found something good, return it
    if best_N is not None and best_count > 0:
        return {'N': best_N, 'pairs': best_pairs}
    
    # Fallback: return a trivial result
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
