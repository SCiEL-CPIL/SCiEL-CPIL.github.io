import time
import math
import numpy as np
from collections import defaultdict

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

    # Strategy: Generate powerful numbers up to a limit M.
    # For each powerful number a, we can form pairs (a, b) where b = N - a.
    # We want to find N that maximizes the number of such pairs.
    # This is equivalent to finding the most frequent sum in {a + b | a, b in P}.
    # However, doing O(P^2) is too slow for large P.
    
    # Alternative approach:
    # We can iterate through powerful numbers a, and for each a, check if N-a is powerful.
    # But we don't know N yet.
    
    # Let's stick to the sum counting method but optimize it.
    # The bottleneck in the previous version was np.bincount with a huge minlength (up to 2*M).
    # If M is large (e.g., 10^7), 2*M is 2*10^7, which is fine for bincount.
    # But P^2 is the real issue. If P ~ 10^5, P^2 = 10^10, which is too slow.
    
    # How many powerful numbers are there up to M?
    # Approximately M^(1/2) + M^(1/3) ... roughly M^(1/2).
    # For M = 10^6, P ~ 1000. P^2 = 10^6. Fast.
    # For M = 10^8, P ~ 10^4. P^2 = 10^8. Maybe 1-2 seconds in Numpy.
    # For M = 10^10, P ~ 10^5. P^2 = 10^10. Too slow.
    
    # So we should try to maximize M such that P^2 is manageable.
    # Let's try M = 10^8 first.
    
    best_result = None
    best_score = 0
    
    limits_to_try = [10**8, 5*10**7, 10**7, 5*10**6]
    
    for M in limits_to_try:
        if time.time() > deadline - 10: # 10s buffer
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            # If L is too large, skip to smaller M
            if L > 20000: # 20000^2 = 4*10^8 ops, might be borderline
                continue
            
            P_np = np.array(P, dtype=np.int64)
            
            # Generate all sums
            # We can do this in chunks to avoid memory blowup if needed, 
            # but for L=20000, L^2 = 4*10^8 int64s = 3.2 GB. This might be too much.
            # Let's reduce the threshold for L.
            # If L=10000, L^2 = 10^8 int64s = 800 MB. Safer.
            
            if L > 10000:
                continue

            sums = np.add.outer(P_np, P_np)
            max_sum_val = sums.max()
            
            # bincount requires int64 indices? No, it takes array of ints.
            # But minlength must be int.
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            d = 0
            if best_N % 2 == 0:
                half_N = best_N // 2
                if half_N in P:
                    d = 1
            
            num_pairs = (C + d) // 2
            
            if num_pairs > best_score:
                best_score = num_pairs
                # Recover pairs
                pairs = []
                P_set = set(P)
                for a in P:
                    b = best_N - a
                    if b >= a:
                        if b in P_set:
                            pairs.append([a, b])
                
                best_result = {'N': best_N, 'pairs': pairs}
                
                # If we found a good result, we might stop early if we are confident?
                # Or just keep looking for better M?
                # Larger M usually gives more pairs.
                # So we break on success? No, we want the best.
                # But if we hit a small M, larger M will likely be better.
                # So we should try largest M first.
                
        except MemoryError:
            continue
        except Exception:
            continue

    if best_result is not None:
        return best_result
    
    # Fallback
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
