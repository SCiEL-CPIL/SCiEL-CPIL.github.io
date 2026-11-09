import itertools
import math
import time

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    max_val = 25
    candidates = list(range(1, max_val + 1))
    
    for size in range(2, 7):
        for combo in itertools.combinations(candidates, size - 1):
            if time.time() > deadline:
                return best_set
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    return best_set
