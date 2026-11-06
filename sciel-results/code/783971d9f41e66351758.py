import time
import random

def _is_conflict(x, S):
    """
    Check if adding x to S creates a forbidden triple {a, b, c} with 1/a = 1/b + 1/c.
    S is a set of integers.
    """
    # Case 1: x is the smallest element (a=x)
    # 1/x = 1/b + 1/c  =>  c = xb / (b-x)
    # b, c must be in S, b != c, b > x, c > x
    for b in S:
        if b <= x:
            continue
        num = x * b
        den = b - x
        if num % den == 0:
            c = num // den
            if c in S and c != b:
                return True

    # Case 2: x is NOT the smallest element.
    # Then x is one of the larger elements (b or c).
    # Due to symmetry, let x be b.
    # 1/a = 1/x + 1/c  =>  1/c = 1/a - 1/x = (x-a)/ax  =>  c = ax / (x-a)
    # a must be in S, a < x.
    # c must be in S, c != a, c != x.
    for a in S:
        if a >= x:
            continue
        num = a * x
        den = x - a
        if num % den == 0:
            c = num // den
            if c in S and c != x and c != a:
                return True

    return False

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initial seed: [91, 180]
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Strategy 1: Strict descending greedy
    S = set()
    for x in range(180, 0, -1):
        if not _is_conflict(x, S):
            S.add(x)
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
        
    # Strategy 2: Randomized restarts
    # We vary the noise level to explore different orderings
    noise_levels = [5, 10, 20, 40, 80]
    restarts_per_level = 50
    
    for noise in noise_levels:
        if time.time() > deadline:
            break
        for _ in range(restarts_per_level):
            if time.time() > deadline:
                break
            
            S = set()
            candidates = list(range(1, 181))
            # Score = -value + noise
            # Higher score -> picked first
            # We want larger values to have higher base score
            scored = [(-x + random.uniform(0, noise), x) for x in candidates]
            scored.sort(key=lambda item: item[0], reverse=True)
            order = [x for _, x in scored]
            
            for x in order:
                if not _is_conflict(x, S):
                    S.add(x)
            
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
                
    return list(best_set)
