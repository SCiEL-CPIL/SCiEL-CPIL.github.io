import time
import math
import numpy as np
from itertools import combinations

DEADLINE = 135.0

def _score(U):
    """Compute score efficiently."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 1.0
    maxU = int(U.max())
    if maxU == 0:
        return 1.0
    # Use set-based computation for small sets, numpy for larger
    if n <= 50:
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(int(a) + int(b))
                diffs.add(int(a) - int(b))
        ns = len(sums)
        nd = len(diffs)
    else:
        sums = np.add.outer(U, U).ravel()
        diffs = np.subtract.outer(U, U).ravel()
        ns = len(np.unique(sums))
        nd = len(np.unique(diffs))
    ratio = nd / ns
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * maxU + 1)

def construct():
    t0 = time.time()
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Generate candidates from structured families
    candidates = set()
    
    # Family: {0} ∪ {r^k}
    for r in range(2, 12):
        for n in range(2, 20):
            U = [0]
            val = 1
            for k in range(n):
                U.append(val)
                val *= r
                if val > 1000000:
                    break
            if max(U) < 1000000:
                candidates.add(tuple(sorted(U)))
    
    # Family: {r^k - 1}
    for r in range(2, 12):
        for n in range(1, 20):
            U = []
            for k in range(n):
                v = r**k - 1
                U.append(v)
                if v > 1000000:
                    break
            if 0 in U and max(U) < 1000000:
                candidates.add(tuple(sorted(U)))
    
    # Small brute force: all subsets of {0,...,M} of size 3-6 containing 0
    for M in range(3, 14):
        elems = list(range(1, M+1))
        for size in range(2, min(6, M+1)):
            for combo in combinations(elems, size):
                U = tuple(sorted([0] + list(combo)))
                candidates.add(U)
    
    # Evaluate candidates
    for U in candidates:
        if time.time() - t0 > DEADLINE:
            break
        s = _score(list(U))
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Local search
    if time.time() - t0 < DEADLINE - 15:
        current = best_set[:]
        cur_score = best_score
        max_iter = 100
        for _ in range(max_iter):
            if time.time() - t0 > DEADLINE - 15:
                break
            improved = False
            maxU = max(current)
            
            # Try adding
            for new_el in range(1, maxU + 3):
                if new_el not in current:
                    U_new = current + [new_el]
                    s = _score(U_new)
                    if s > cur_score + 1e-12:
                        current = U_new
                        cur_score = s
                        improved = True
                        break
            
            # Try removing
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    U_new = current[:i] + current[i+1:]
                    if len(U_new) < 2:
                        continue
                    s = _score(U_new)
                    if s > cur_score + 1e-12:
                        current = U_new
                        cur_score = s
                        improved = True
                        break
            
            # Try swapping
            if not improved:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(1, maxU + 2):
                        if new_el not in current:
                            U_new = current[:]
                            U_new[i] = new_el
                            s = _score(U_new)
                            if s > cur_score + 1e-12:
                                current = U_new
                                cur_score = s
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        if cur_score > best_score:
            best_score = cur_score
            best_set = current
    
    return best_set
