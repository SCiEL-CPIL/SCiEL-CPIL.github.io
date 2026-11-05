"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _verify(arr):
    """Check that all 2^n subset sums are distinct and arr contains exactly n distinct positive ints."""
    n = len(arr)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += arr[i]
        if s in sums:
            return False
        sums.add(s)
    if len(sums) != 1 << n:
        return False
    if len(set(arr)) != n:
        return False
    if any(a <= 0 for a in arr):
        return False
    return True

def _greedy(n, deadline):
    """Greedy: pick smallest possible next element."""
    arr = []
    for i in range(n):
        if time.time() > deadline:
            break
        for candidate in range(1, 1 << 20):
            if time.time() > deadline:
                break
            trial = arr + [candidate]
            # Quick check: all subset sums of trial must be distinct
            ok = True
            sums = set()
            for mask in range(1 << len(trial)):
                s = 0
                for j in range(len(trial)):
                    if mask >> j & 1:
                        s += trial[j]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if ok:
                arr.append(candidate)
                break
    return arr

def construct():
    deadline = time.time() + 80  # safety margin
    n = 12
    
    # The powers of 2 give max element 2^11 = 2048
    best = [1 << i for i in range(n)]
    best_val = best[-1]
    
    # Try to find a solution with max element < 2048
    # Key insight: if a_1 < a_2 < ... < a_n and all subset sums are distinct,
    # then a_i > sum of all previous a_j for each i (this is sufficient but not necessary)
    # Actually, the necessary and sufficient condition is just that all subset sums are distinct.
    # But a stronger sufficient condition: a_i > sum(a_1..a_{i-1}) ensures distinct subset sums.
    
    # Let's try a DFS/search approach to find any valid set with max < 2048.
    # The space is small enough if we use the "a_i > sum(previous)" sufficient condition first,
    # then try relaxing it.
    
    # First, try with the sufficient condition a_i > sum(previous):
    # This gives: a_1 >= 1, a_2 > a_1, a_3 > a_1+a_2, ...
    # Minimum sequence: 1, 2, 4, 8, ..., 2^11 = 2048. So this condition alone gives 2048.
    
    # But we can do better by not requiring a_i > sum(previous), only requiring distinct subset sums.
    # Let's do a search.
    
    def search(arr, max_allowed):
        """DFS to extend arr to length 12, with all elements <= max_allowed."""
        if len(arr) == n:
            return arr[:]
        if time.time() > deadline:
            return None
        # Try candidates from 1 to max_allowed
        for cand in range(1, max_allowed + 1):
            if time.time() > deadline:
                return None
            trial = arr + [cand]
            # Check all subset sums of trial are distinct
            sums = set()
            ok = True
            for mask in range(1 << len(trial)):
                s = 0
                for j in range(len(trial)):
                    if mask >> j & 1:
                        s += trial[j]
                if s in sums:
                    ok = False
                    break
                sums.add(s)
            if ok:
                result = search(trial, max_allowed)
                if result is not None:
                    return result
        return None
    
    # Try max_allowed values from 2047 down to some lower bound
    # Lower bound: we need 4096 distinct subset sums in [0, sum(arr)]
    # so sum(arr) >= 4095. With 12 elements each <= M, sum <= 12M, so M >= 4095/12 ~ 342.
    # But realistically M must be larger.
    
    # Try a few values of max_allowed
    for max_allowed in [2047, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000]:
        if time.time() > deadline:
            break
        result = search([], max_allowed)
        if result is not None and len(result) == n:
            if max(result) < best_val:
                best = result
                best_val = max(result)
    
    return best
# EVOLVE-BLOCK-END
