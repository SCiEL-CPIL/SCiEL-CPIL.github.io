"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict
from bisect import bisect_left, bisect_right
import numpy as np

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form with sieve-based square-free check."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    # Precompute square-free status using sieve-like approach
    if max_b >= 1:
        sq_free = [True] * (max_b + 1)
        if max_b >= 0:
            sq_free[0] = False
        if max_b >= 1:
            sq_free[1] = True
        for p in range(2, int(math.isqrt(max_b)) + 1):
            if sq_free[p]:
                p2 = p * p
                for m in range(p2, max_b + 1, p2):
                    sq_free[m] = False
    
    for b in range(1, max_b + 1):
        if b > max_b:
            break
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def _count_pairs_exact(powerful, powerful_set, N):
    """Count pairs (a, b) with a <= b, a + b = N, both powerful. Exact."""
    count = 0
    for a in powerful:
        if a > N // 2:
            break
        b = N - a
        if b >= a and b in powerful_set:
            count += 1
    return count

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    start_time = time.time()
    
    # Phase 1: Generate powerful numbers
    limit = 50_000_000  # 5 * 10^7, manageable for exact counting
    
    powerful = _generate_powerful(limit)
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Exact pairwise sum counting using chunked approach
    # For each powerful number a, compute a + b for all b in powerful where b >= a
    # Accumulate counts in a dictionary
    
    sum_counts = defaultdict(int)
    
    # Convert to list for faster access
    pw_list = powerful
    K = len(pw_list)
    
    # Process in chunks to stay within time
    chunk_size = 2000
    for i_start in range(0, K, chunk_size):
        i_end = min(i_start + chunk_size, K)
        # For each a in pw_list[i_start:i_end], add to all b >= a
        for i in range(i_start, i_end):
            a = pw_list[i]
            # b must be >= a and b = N - a, but we're counting all sums
            # For each b in pw_list[i:] (b >= a), sum = a + b
            for j in range(i, K):
                b = pw_list[j]
                s = a + b
                sum_counts[s] += 1
        
        # Time check
        if time.time() > deadline - 30:
            break
    
    # Find the best N
    if not sum_counts:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Get top candidates
    sorted_sums = sorted(sum_counts.items(), key=lambda x: -x[1])
    
    best_N = None
    best_count = 0
    
    for N, cnt in sorted_sums[:50]:  # Check top 50 candidates
        if N < 2:
            continue
        # Verify exact count
        actual_count = _count_pairs_exact(powerful, powerful_set, N)
        if actual_count > best_count:
            best_count = actual_count
            best_N = N
            if best_count >= 10:  # Early termination if we find a good one
                break
    
    if best_N is None or best_count <= 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Generate the actual pairs for best_N
    pairs = []
    for a in powerful:
        if a > best_N // 2:
            break
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    # Verify
    if len(pairs) != best_count:
        best_count = len(pairs)
    
    # If we have time, try a larger limit
    remaining = deadline - time.time()
    if remaining > 40 and best_count < 15:
        limit2 = 80_000_000
        powerful2 = _generate_powerful(limit2)
        powerful_set2 = set(powerful2)
        K2 = len(powerful2)
        
        if K2 > K:
            sum_counts2 = defaultdict(int)
            pw_list2 = powerful2
            chunk_size2 = 1500
            for i_start in range(0, K2, chunk_size2):
                i_end = min(i_start + chunk_size2, K2)
                for i in range(i_start, i_end):
                    a = pw_list2[i]
                    for j in range(i, K2):
                        b = pw_list2[j]
                        s = a + b
                        sum_counts2[s] += 1
                if time.time() > deadline - 20:
                    break
            
            if sum_counts2:
                sorted_sums2 = sorted(sum_counts2.items(), key=lambda x: -x[1])
                for N2, cnt2 in sorted_sums2[:50]:
                    if N2 < 2:
                        continue
                    actual_count2 = _count_pairs_exact(powerful2, powerful_set2, N2)
                    if actual_count2 > best_count:
                        best_count = actual_count2
                        best_N = N2
                        powerful = powerful2
                        powerful_set = powerful_set2
                        if best_count >= 15:
                            break
            
            # Regenerate pairs if we found a better N
            pairs = []
            for a in powerful:
                if a > best_N // 2:
                    break
                b = best_N - a
                if b >= a and b in powerful_set:
                    pairs.append([a, b])
            if len(pairs) != best_count:
                best_count = len(pairs)
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
