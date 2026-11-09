import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        # Precompute square-free v
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

    # Try different limits, starting high
    # Use int32 to save memory, allowing larger M
    limits_to_try = [800_000_000, 500_000_000, 200_000_000]
    
    P = []
    max_sum = 0
    
    for M in limits_to_try:
        if time.time() > deadline:
            break
            
        try:
            P = generate_powerful(M)
            if len(P) < 100:
                continue
                
            L = len(P)
            # Estimate memory
            # L * L * 4 bytes (int32)
            mem_needed = L * L * 4
            # 8GB limit check?
            if mem_needed > 8 * 1024**3:
                continue
            
            # Convert to numpy array
            # Check if 2*M fits in int32
            if 2 * M <= 2147483647:
                P_np = np.array(P, dtype=np.int32)
                sums = np.add.outer(P_np, P_np)
                max_sum_val = int(sums.max())
                counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            else:
                P_np = np.array(P, dtype=np.int64)
                sums = np.add.outer(P_np, P_np)
                max_sum_val = int(sums.max())
                counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            # Find the sum with max count
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            # Calculate distinct pairs count
            # C = 2 * (a < b) + 1 * (a == b)
            # We want (a < b) + (a == b)
            # Let d = 1 if N is even and N/2 is powerful
            d = 0
            if best_N % 2 == 0:
                half_N = best_N // 2
                # Check if half_N is in P
                # P is sorted, use binary search or set
                # For speed, use set? Or since P is numpy, searchsorted?
                # We already have P as list.
                if half_N in P: # O(log L) if we use bisect, but 'in' on list is O(L). 
                               # L is ~40k. O(L) is 40k ops. Fine.
                    d = 1
            
            num_pairs = (C + d) // 2
            
            # Recover pairs
            pairs = []
            P_set = set(P)
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            return {'N': best_N, 'pairs': pairs}
            
        except MemoryError:
            # Try next smaller limit
            continue
        except Exception:
            continue

    # Fallback if all failed or time ran out
    # Use a very small limit
    try:
        P = generate_powerful(1000)
        P_np = np.array(P, dtype=np.int64)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=sums.max() + 1)
        best_N = int(np.argmax(counts))
        C = int(counts[best_N])
        d = 1 if (best_N % 2 == 0 and best_N // 2 in P) else 0
        num_pairs = (C + d) // 2
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
