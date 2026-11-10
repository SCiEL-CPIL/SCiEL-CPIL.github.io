import time
import itertools
import random

_DEADLINE = 55  # seconds, safety margin under 60s
_START = time.time()

def _check_validity(arr):
    """Check that no two distinct non-empty subset sums divide one another."""
    n = len(arr)
    if n > 20:
        # Too many subsets to enumerate; use heuristic or skip
        return False
    sums = set()
    for r in range(1, n + 1):
        for comb in itertools.combinations(arr, r):
            sums.add(sum(comb))
    sums = sorted(sums)
    for i in range(len(sums)):
        a = sums[i]
        for j in range(i + 1, len(sums)):
            b = sums[j]
            if b % a == 0:
                return False
    return True

def _incremental_check(arr, candidate):
    """Check if adding candidate to arr maintains the property.
    Returns True if valid, False otherwise."""
    n = len(arr) + 1
    if n > 20:
        return False
    # Compute all subset sums of arr
    arr_sums = {0}
    for x in arr:
        arr_sums = arr_sums | {s + x for s in arr_sums}
    
    # New sums = old sums ∪ {s + candidate for s in arr_sums}
    new_sums = set(arr_sums)
    for s in arr_sums:
        new_sums.add(s + candidate)
    
    # Remove 0 (we only care about non-empty subset sums)
    new_sums.discard(0)
    
    # Check divisibility among all pairs
    new_sums_list = sorted(new_sums)
    for i in range(len(new_sums_list)):
        a = new_sums_list[i]
        for j in range(i + 1, len(new_sums_list)):
            b = new_sums_list[j]
            if b % a == 0:
                return False
    return True

def _greedy_build(pool, start_size=0, max_size=15, deadline=None):
    """Greedily build a valid set from pool."""
    if deadline is None:
        deadline = _DEADLINE
    current = []
    sums_set = {0}
    
    for x in pool:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        
        # Try adding x
        # Check if any new sum divides an existing sum or vice versa
        # New sums are {s + x for s in sums_set}
        new_sums = set()
        for s in sums_set:
            new_sums.add(s + x)
        
        # Check all pairs between old sums and new sums, and new sums among themselves
        valid = True
        old_sums_list = list(sums_set)
        new_sums_list = list(new_sums)
        
        # Check old vs new
        for os in old_sums_list:
            if os == 0:
                continue
            for ns in new_sums_list:
                if ns == 0:
                    continue
                if ns % os == 0 or (os % ns == 0):
                    valid = False
                    break
            if not valid:
                break
        
        # Check new vs new (distinct sums)
        if valid:
            seen = set()
            for ns in new_sums_list:
                if ns in seen:
                    continue
                seen.add(ns)
                for ns2 in new_sums_list:
                    if ns2 <= ns:
                        continue
                    if ns2 % ns == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current.append(x)
            sums_set = sums_set | new_sums
    
    return current

def construct():
    global _START
    if time.time() - _START > _DEADLINE:
        return [31, 32, 34]
    
    best = [31, 32, 34]
    
    # Strategy: Try different pool configurations
    # Key insight: to maximize size, we want elements that are "incomparable"
    # in terms of subset sums. Using numbers near powers of 2 or in specific ranges.
    
    random.seed(42)
    
    # Pool 1: Numbers in range [100, 200]
    pool1 = list(range(100, 200))
    random.shuffle(pool1)
    result = _greedy_build(pool1[:30], max_size=15)
    if len(result) > len(best) and _check_validity(result):
        best = result
    
    if time.time() - _START > _DEADLINE - 5:
        return best
    
    # Pool 2: Numbers in range [150, 256]
    pool2 = list(range(150, 257))
    random.shuffle(pool2)
    result = _greedy_build(pool2[:30], max_size=15)
    if len(result) > len(best) and _check_validity(result):
        best = result
    
    if time.time() - _START > _DEADLINE - 5:
        return best
    
    # Pool 3: Mix of larger numbers
    pool3 = [random.randint(80, 256) for _ in range(40)]
    pool3 = sorted(set(pool3))
    random.shuffle(pool3)
    result = _greedy_build(pool3[:30], max_size=15)
    if len(result) > len(best) and _check_validity(result):
        best = result
    
    if time.time() - _START > _DEADLINE - 5:
        return best
    
    # Try specific known-good starting points and expand
    # Start with numbers that are pairwise incomparable in simple ways
    base_candidates = [
        [100, 150, 200, 220, 240],
        [120, 160, 190, 210, 230],
        [110, 145, 175, 205, 235],
    ]
    
    for base in base_candidates:
        if time.time() - _START > _DEADLINE - 3:
            break
        # Try to extend base
        current = base[:]
        pool = [x for x in range(50, 257) if x not in current]
        random.shuffle(pool)
        result = _greedy_build(pool[:25], max_size=15, deadline=_START + _DEADLINE - 2)
        # Merge: start from base, try adding from pool
        current = base[:]
        for x in pool:
            if time.time() > _START + _DEADLINE - 2:
                break
            if len(current) >= 15:
                break
            if _incremental_check(current, x):
                current.append(x)
        if len(current) > len(best) and _check_validity(current):
            best = current
    
    return [int(x) for x in best]
