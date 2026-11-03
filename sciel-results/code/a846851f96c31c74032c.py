import time
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums_set = set()
    diffs_set = set()
    for a in U:
        for b in U:
            sums_set.add(a + b)
            diffs_set.add(a - b)
    
    ratio = len(diffs_set) / len(sums_set)
    if ratio <= 1:
        return -1
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 105
    
    best_set = [0, 1, 2, 9]
    best_score = score_set(best_set)
    
    # Search for best 4-element set {0, 1, b, c}
    for b in range(2, 20):
        for c in range(b + 1, 50):
            if time.time() > deadline:
                break
            U = [0, 1, b, c]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
        
    # Search for best 5-element set {0, 1, 2, b, c}
    for b in range(3, 20):
        for c in range(b + 1, 100):
            if time.time() > deadline:
                break
            U = [0, 1, 2, b, c]
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
                
    return best_set
