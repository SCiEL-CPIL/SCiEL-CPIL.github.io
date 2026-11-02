import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = 1.0
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        size_s = len(sums)
        size_d = len(diffs)
        
        if size_s == 0:
            return 0.0
            
        ratio = size_d / size_s
        if ratio <= 1.0:
            # If ratio <= 1, log is <= 0, score <= 1
            # We want > 1, so we need ratio > 1
            return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
        
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    # Try small sets
    # We can try all subsets of {0, 1, ..., K} of size N
    # But N and K are unknown.
    # Let's try a few heuristics.
    
    # Heuristic 1: Geometric Progressions
    for r in range(2, 10):
        for n in range(3, 8):
            U = [r**i for i in range(n)]
            if 0 not in U:
                U.append(0)
            U = sorted(set(U))
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best_set = U
            if time.time() > deadline:
                return best_set

    # Heuristic 2: Random sets with small max
    import random
    random.seed(42)
    
    for n in range(3, 12):
        for max_val in range(n, 50):
            for _ in range(1000):
                if time.time() > deadline:
                    return best_set
                # Generate n-1 random numbers from 1..max_val
                candidates = list(range(1, max_val + 1))
                if len(candidates) < n - 1:
                    continue
                sample = random.sample(candidates, n - 1)
                U = sorted([0] + sample)
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # Heuristic 3: Exhaustive search for very small sets
    # n=3, 4, 5 with small max
    for n in range(3, 6):
        for max_val in range(n, 20):
            # Generate all combinations of n-1 elements from 1..max_val
            range_vals = list(range(1, max_val + 1))
            if len(range_vals) < n - 1:
                continue
            for comb in itertools.combinations(range_vals, n - 1):
                if time.time() > deadline:
                    return best_set
                U = sorted([0] + list(comb))
                s = calc_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    return best_set
