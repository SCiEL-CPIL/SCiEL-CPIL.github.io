"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
from bisect import bisect_left, bisect_right
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    max_b = int(round(limit ** (1/3))) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        # Check if b is square-free
        temp_b = b
        d = 2
        sq_free = True
        while d * d <= temp_b:
            if temp_b % (d * d) == 0:
                sq_free = False
                break
            d += 1 if d == 2 else 2
        if not sq_free:
            continue
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers up to a large limit
    limit = 60_000_000  # 6 * 10^7
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Count representations using efficient numpy-based approach
    # Key insight: instead of O(K^2) pair enumeration, use the fact that
    # for each a, we need to find b = N - a that is also powerful.
    # We'll use a different strategy: for each candidate N (sum of two powerfuls),
    # count how many pairs sum to it.
    
    # More efficient: iterate over a values, and for each a, the possible sums
    # are a + b for b in powerful with b >= a. Use numpy vectorized ops.
    
    arr = np.array(powerful, dtype=np.int64)
    
    # We'll accumulate counts using a dictionary, but process in batches
    # to avoid O(K^2) Python loops.
    
    sum_counts = defaultdict(int)
    
    # Process a values in order; for each a, find valid b range
    # Use numpy to compute all sums for a given a at once
    processed = 0
    for i in range(K):
        a = arr[i]
        if a > limit // 2:
            break
        # b must be >= a and <= limit - a
        j_start = bisect_left(arr, a)
        j_end = bisect_right(arr, limit - a)
        if j_start >= j_end:
            continue
        
        # Get the b values and compute sums
        b_vals = arr[j_start:j_end]
        sums = a + b_vals
        
        # Use np.unique to count occurrences of each sum
        unique_sums, counts = np.unique(sums, return_counts=True)
        for s, c in zip(unique_sums, counts):
            sum_counts[int(s)] += int(c)
        
        processed += 1
        
        # Time check: if we're running low, stop
        if time.time() > deadline - 40:
            break
    
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find the best N
    best_N = max(sum_counts.keys(), key=lambda n: sum_counts[n])
    best_count = sum_counts[best_N]
    
    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set and b >= a:
            pairs.append([a, b])
    
    # Verify
    if len(pairs) != best_count:
        # If mismatch, just return what we have (shouldn't happen)
        best_count = len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
