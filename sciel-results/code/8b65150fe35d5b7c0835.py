import math
from itertools import combinations

def score_set(U):
    U = sorted(U)
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
            
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
    
    # Search for small N and small Max
    # N is size of U
    # We include 0, so we pick N-1 from 1..Max
    for N in range(3, 9):
        # Max value to search
        # Heuristic: Sidon sets have max ~ N^2
        # Let's go up to 40 for safety
        max_val = 40
        for combo in combinations(range(1, max_val + 1), N - 1):
            U = [0] + list(combo)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_set
