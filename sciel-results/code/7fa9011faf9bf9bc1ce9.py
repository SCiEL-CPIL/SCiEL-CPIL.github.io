import time
import math
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_U = [0, 1, 3, 6, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Exhaustive search over subsets of {1,...,30} of size 3-6
    pool = list(range(1, 31))
    
    for size in [3, 4, 5, 6]:
        for combo in combinations(pool, size):
            if time.time() > deadline:
                return best_U
            U = [0] + list(combo)
            try_set(U)
    
    # Size 7 with smaller pool
    pool7 = list(range(1, 20))
    for combo in combinations(pool7, 7):
        if time.time() > deadline:
            return best_U
        U = [0] + list(combo)
        try_set(U)
    
    return best_U
