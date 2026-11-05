"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _subset_sum_exists(elements, k, target):
    """Check if there's a subset of exactly k elements from elements summing to target."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if k == n:
        return sum(elements) == target
    
    sorted_e = sorted(elements)
    min_s = sum(sorted_e[:k])
    max_s = sum(sorted_e[-k:])
    if target < min_s or target > max_s:
        return False
    
    # Use complement if beneficial
    ck = n - k
    if ck < k:
        ct = sum(elements) - target
        return _subset_sum_exists(elements, ck, ct)
    
    if k == 1:
        return target in elements
    if k == 2:
        seen = set()
        for x in elements:
            if (target - x) in seen:
                return True
            seen.add(x)
        return False
    if k == 3:
        es = set(elements)
        for i in range(n):
            for j in range(i+1, n):
                z = target - elements[i] - elements[j]
                if z in es and z != elements[i] and z != elements[j]:
                    return True
        return False
    if k <= 5:
        for comb in combinations(elements, k):
            if sum(comb) == target:
                return True
        return False
    
    # Fallback: combinations (expensive but correct)
    for comb in combinations(elements, k):
        if sum(comb) == target:
            return True
    return False


def _valid_add(selected, cand):
    """Check if adding cand to selected keeps the set valid."""
    n = len(selected)
    # Check cand is not mean of any subset of >=2 from selected
    if n >= 2:
        for k in range(2, n+1):
            if _subset_sum_exists(selected, k, k*cand):
                return False
    # Check no a in selected is mean of subset including cand
    for a in selected:
        others = [x for x in selected if x != a]
        m = len(others)
        for k in range(1, m+1):
            if _subset_sum_exists(others, k, (k+1)*a - cand):
                return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Build candidate pool: multiplicative structure + full range
    cands = set()
    for i in range(10):
        for j in range(10):
            v = (1 << i) * (3 ** j)
            if 1 <= v <= 512:
                cands.add(v)
    for v in range(1, 513):
        cands.add(v)
    
    clist = sorted(cands)
    
    best = []
    best_sz = 0
    
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        shuf = clist[:]
        random.shuffle(shuf)
        
        cur = []
        for c in shuf:
            if time.time() > deadline - 3:
                break
            if _valid_add(cur, c):
                cur.append(c)
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    if best_sz < 48:
        rem = [x for x in range(1, 513) if x not in best]
        random.shuffle(rem)
        for c in rem:
            if time.time() > deadline - 2:
                break
            if _valid_add(best, c):
                best.append(c)
                if len(best) >= 48:
                    break
    
    return best[:48]
# EVOLVE-BLOCK-END
