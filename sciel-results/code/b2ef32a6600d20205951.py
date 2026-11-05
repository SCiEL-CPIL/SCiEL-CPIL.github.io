import time
import random

def _subset_sum_exists(elements, k, target):
    """Check if there's a subset of exactly k elements from elements summing to target.
    Returns True if such a subset exists."""
    n = len(elements)
    if k < 0 or k > n:
        return False
    if k == 0:
        return target == 0
    if k == n:
        return sum(elements) == target
    if target < 0:
        return False
    
    # Early bounds
    sorted_e = sorted(elements)
    min_s = sum(sorted_e[:k])
    max_s = sum(sorted_e[-k:])
    if target < min_s or target > max_s:
        return False
    
    # Use complement if beneficial
    ck = n - k
    if ck < k:
        ct = sum(elements) - target
        if ct < 0:
            return False
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
    
    # DP: sums[j] = set of sums achievable with j elements
    sums = [set() for _ in range(k+1)]
    sums[0].add(0)
    
    for e in elements:
        for j in range(k, 0, -1):
            for s in list(sums[j-1]):
                new_s = s + e
                if new_s <= target:
                    sums[j].add(new_s)
    
    return target in sums[k]

def _valid_add(S, c):
    """Check if c can be added to set S without violating the mean condition.
    
    Condition 1: c is not the mean of any subset of S of size >= 2
    Condition 2: no x in S is the mean of a subset of (S ∪ {c}) \ {x} of size >= 2 that includes c
    """
    S = list(S)
    n = len(S)
    
    # Condition 1: c is not the mean of any subset of S of size >= 2
    for k in range(2, n+1):
        if _subset_sum_exists(S, k, k * c):
            return False
    
    # Condition 2: for each x in S, no subset of (S \ {x}) of size k-1 (k>=2)
    # has sum = k*x - c
    for x in S:
        others = [y for y in S if y != x]
        for k in range(2, n+1):
            target = k * x - c
            if target < 0:
                continue
            if _subset_sum_exists(others, k-1, target):
                return False
    
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    
    best = []
    best_sz = 0
    
    # Try multiple random orderings with greedy construction
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        cands = list(range(1, 513))
        random.shuffle(cands)
        
        cur = []
        for c in cands:
            if time.time() > deadline - 5:
                break
            if _valid_add(cur, c):
                cur.append(c)
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    # Local search: try to swap elements to increase size
    if best_sz < 48 and time.time() < deadline - 5:
        cur = best[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(cur)):
                if time.time() > deadline - 5:
                    break
                for c in range(1, 513):
                    if c in cur:
                        continue
                    new_cur = cur[:i] + cur[i+1:]
                    if _valid_add(new_cur, c):
                        cur = new_cur + [c]
                        improved = True
                        break
                if improved:
                    break
        
        if len(cur) > best_sz:
            best_sz = len(cur)
            best = cur[:]
            if best_sz >= 48:
                break
    
    return best[:48]
