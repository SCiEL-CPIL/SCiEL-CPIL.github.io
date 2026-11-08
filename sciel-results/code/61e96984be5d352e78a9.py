import time
import math
import itertools

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    maxU = max(U)
    if maxU == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ps = len(sumset)
    pd = len(diffset)
    if ps == 0:
        return -1
    ratio = pd / ps
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * maxU + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_score = -1
    best_set = [0, 1, 3]
    
    s = score_set([0, 1, 3])
    if s > best_score:
        best_score = s
        best_set = [0, 1, 3]
    
    # Exhaustive search over subsets of {1,...,K} for small K
    for K in range(3, 18):
        if time.time() > deadline:
            break
        for n in range(3, min(7, K + 1)):
            if time.time() > deadline:
                break
            from math import comb
            if comb(K, n - 1) > 50000:
                continue
            for combo in itertools.combinations(range(1, K + 1), n - 1):
                if time.time() > deadline:
                    break
                U = [0] + list(combo)
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Try geometric sequences
    for base in [2, 3, 4, 5]:
        if time.time() > deadline:
            break
        for length in range(3, 9):
            U = [0]
            cur = 1
            for i in range(length - 1):
                U.append(cur)
                cur *= base
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    return best_set
