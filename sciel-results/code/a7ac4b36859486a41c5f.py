import math
import time
import numpy as np

def construct():
    # Deadline guard
    START_TIME = time.time()
    DEADLINE = 140  # seconds
    
    def compute_score(u):
        u = sorted(set(u))
        if not u or 0 not in u:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        arr = np.array(u, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        sumset_size = len(np.unique(sums))
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        return 1 + math.log(r) / math.log(m)
    
    def is_golomb(u):
        u = sorted(u)
        diffs = set()
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                d = u[j] - u[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    # Known good starting points (Golomb rulers)
    # We want small max_u and high diff/sum ratio.
    candidates = [
        [0, 1, 3],      # n=3
        [0, 1, 4, 6],   # n=4
        [0, 1, 4, 9, 11], # n=5
        [0, 1, 4, 10, 12, 17], # n=6
        [0, 2, 5, 7],   # Check if this is Golomb? 2,3,5,7,5,3 -> No.
        [0, 1, 5, 7],   # Golomb
    ]
    
    # Filter valid Golomb rulers and compute scores
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    for u in candidates:
        if not is_golomb(u):
            continue
        s = compute_score(u)
        if s > best_score:
            best_score = s
            best_set = u
            
    # Local search to improve the best candidate
    # Try to reduce sumset size by swapping elements
    # This is expensive, so we limit it.
    
    def local_search(base_u, time_limit):
        start_time = time.time()
        best_u = list(base_u)
        best_score_local = compute_score(best_u)
        
        # Try swapping each element (except 0) with nearby values
        # to see if sumset size decreases while maintaining Golomb property
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            current_max = max(best_u)
            n = len(best_u)
            # Search range for new values
            search_hi = current_max + 20
            search_lo = 1
            
            # Generate list of indices to try swapping
            indices_to_try = list(range(1, n))
            np.random.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                if (time.time() - start_time) > time_limit:
                    break
                old_val = best_u[idx]
                # Try replacing with values in a window
                # Window size depends on n
                window = max(5, n * 2)
                for new_val in range(max(search_lo, old_val - window), min(search_hi, old_val + window) + 1):
                    if new_val == old_val:
                        continue
                    if new_val in best_u:
                        continue
                    
                    candidate = sorted([x if x != old_val else new_val for x in best_u])
                    if not is_golomb(candidate):
                        continue
                    
                    s = compute_score(candidate)
                    if s > best_score_local + 1e-9:
                        best_score_local = s
                        best_u = candidate
                        improved = True
                        break # Restart search with new best
                    
        return best_u, best_score_local

    # Allocate remaining time to local search
    elapsed = time.time() - START_TIME
    time_remaining = DEADLINE - elapsed - 5  # 5s safety margin
    if time_remaining > 5:
        ext_u, ext_score = local_search(best_set, time_remaining)
        if ext_score > best_score:
            best_set = ext_u
            best_score = ext_score
            
    return best_set
