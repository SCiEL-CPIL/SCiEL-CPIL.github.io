import itertools
import math
import time

def calculate_score(U):
    if not U or 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
            
    num = math.log(len(diff_set) / len(sum_set))
    return 1 + num / denom

def construct():
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Search for small sets
    # We assume 0 is always in the set.
    # We search for subsets of {1, ..., M}
    max_m = 20
    deadline = time.time() + 100 # Leave 20s margin
    
    for m in range(1, max_m + 1):
        if time.time() > deadline:
            break
        # Try subsets of size 1 to 6 of {1, ..., m}
        # Since 0 is fixed, we pick k elements from 1..m
        for k in range(1, 7):
            if time.time() > deadline:
                break
            for subset in itertools.combinations(range(1, m + 1), k):
                U = [0] + list(subset)
                score = calculate_score(U)
                if score > best_score:
                    best_score = score
                    best_set = U
    
    return best_set
