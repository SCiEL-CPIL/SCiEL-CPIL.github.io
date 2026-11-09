import time
import math
import numpy as np
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 150s kill

    def is_square_free(n):
        if n < 2:
            return True
        r = int(math.isqrt(n))
        for p in range(2, r + 1):
            if n % (p * p) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 2
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
                
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Strategy: Try multiple limits, but be smarter about finding good N
    # Key insight: We want N that has many representations. 
    # Instead of finding max count over ALL sums, we can try targeted approaches.
    
    best_result = None
    best_count = 0
    
    # Try a range of limits
    limits_to_try = [300_000_000, 200_000_000, 150_000_000, 100_000_000]
    
    for M in limits_to_try:
        if time.time() > deadline - 5:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            if L < 50:
                continue
                
            # Check if we have time for this computation
            # Outer product is O(L^2) operations
            if L > 20000:
                # Too large, skip to avoid timeout
                continue
                
            P_np = np.array(P, dtype=np.int64)
            
            # Compute all pairwise sums efficiently
            # Use chunked approach if needed
            sums = np.add.outer(P_np, P_np)
            max_sum_val = int(sums.max())
            
            # Count occurrences - this is the bottleneck
            # For large sums, bincount can be memory intensive
            # Use a dictionary-based approach for very large ranges
            if max_sum_val > 500_000_000:
                # Fallback to dict counting
                from collections import Counter
                sum_counts = Counter()
                flat_sums = sums.ravel()
                for s in flat_sums:
                    sum_counts[int(s)] += 1
                if not sum_counts:
                    continue
                best_N = max(sum_counts, key=sum_counts.get)
                C = sum_counts[best_N]
            else:
                counts = np.bincount(flat_sums, minlength=max_sum_val + 1)
                max_count_idx = np.argmax(counts)
                best_N = int(max_count_idx)
                C = int(counts[best_N])
            
            # Calculate distinct pairs
            P_set = set(P)
            pairs = []
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            num_pairs = len(pairs)
            
            if num_pairs > best_count:
                best_count = num_pairs
                best_result = {'N': best_N, 'pairs': pairs}
                
            # If we found a good result, we can stop
            if num_pairs >= 10:
                break
                
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result is not None:
        return best_result
    
    # Fallback: use a small limit
    try:
        P = generate_powerful(5000)
        P_np = np.array(P, dtype=np.int64)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=int(sums.max()) + 1)
        best_N = int(np.argmax(counts))
        P_set = set(P)
        pairs = []
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
