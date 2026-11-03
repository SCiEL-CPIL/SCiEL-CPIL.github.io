import time
import itertools
from math import log

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 110  # 110s deadline, 40s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try structured families
    # Family 1: {0, 1, 3, 7, ..., 2^k - 1}
    for k in range(2, 20):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 2: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 3: Mixed - try to grow from best
    # Local search: try adding elements
    def try_grow(U, max_val_limit=10000):
        U = sorted(set(U))
        current_score = compute_score(U)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate
            candidates = list(range(1, min(max_val_limit, max(U)*2+10)))
            # Also try some larger candidates
            candidates += [max(U)*2, max(U)*3, max(U)*5, max(U)*10]
            candidates = sorted(set(candidates))
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_score = new_score
                    U = new_U
                    improved = True
                    break  # Restart candidate search with new U
        return U, current_score
    
    # Try growing from a few starting points
    starts = [[0,1,3], [0,1,3,7], [0,1,3,9]]
    for start_U in starts:
        if time.time() > deadline:
            break
        U, s = try_grow(start_U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U
