import itertools
import math
import time

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    sum_set = {a + b for a in U for b in U}
    diff_set = {a - b for a in U for b in U}
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Known good candidate
    candidate = [0, 1, 3, 9]
    s = score_set(candidate)
    if s > best_score:
        best_score = s
        best_set = candidate

    # Search small sets
    # Try sizes up to 5, max element up to 20
    # This is a heuristic search
    
    max_size = 5
    max_val = 20
    
    # Generate combinations
    # To keep it fast, we only consider sets containing 0
    # and sorted
    
    # Precompute scores for small sets?
    # Just iterate
    
    for size in range(2, max_size + 1):
        # Choose size-1 other elements from 1..max_val
        for combo in itertools.combinations(range(1, max_val + 1), size - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
        if time.time() > deadline:
            break
    
    return best_set
