import math
from itertools import combinations

def score_set(U):
    U = sorted(U)
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    
    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Exhaustive search for small N and small Max
    # N=3
    # Max up to 10
    for max_u in range(3, 15):
        # Pick 2 elements from 1..max_u-1
        for combo in combinations(range(1, max_u), 2):
            U = [0] + list(combo)
            # Ensure max is actually max_u? No, we just iterate over all subsets of range(1, max_u)
            # But this generates duplicates if we don't fix max.
            # Better: iterate max_u, pick N-1 from 1..max_u-1, one must be max_u-1? No.
            # Just iterate all subsets of size N from 1..MAX_VAL, include 0.
            
    # Let's do it properly:
    MAX_VAL = 25
    for N in range(3, 7):
        # Pick N-1 numbers from 1 to MAX_VAL
        # To avoid duplicates, we can just generate combinations
        for combo in combinations(range(1, MAX_VAL + 1), N - 1):
            U = [0] + list(combo)
            # Optimization: if max(U) is small, it's more likely to be good
            # But we check all.
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_set
