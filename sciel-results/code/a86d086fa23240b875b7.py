import time
from itertools import combinations
from fractions import Fraction

def _check_valid(S):
    """Check if S is reciprocal-subsum-free using exact rational arithmetic."""
    S_set = set(S)
    if not S_set:
        return True
    for x in S_set:
        others = [y for y in S_set if y != x]
        target = Fraction(1, x)
        # Check subsets of others
        # We limit the size of subsets we check.
        # If sum of reciprocals of any k elements > target, larger subsets won't help
        # (since reciprocals are positive).
        # Actually, we just need to check if ANY subset sums to target.
        # Meet-in-the-middle is better for large sets, but for small sets (<= 24)
        # brute force with pruning or simple DP might be okay.
        # However, the current program uses combinations up to size 6.
        # Let's stick to a robust but fast check.
        # Since max size is 24, and we're checking subsets of others (max 23),
        # 2^23 is too big.
        # But we can use the property that 1/x is small for large x.
        # For x >= 13, 1/x <= 1/13.
        # The sum of reciprocals of any subset of {1..24} \ {x} that equals 1/x.
        # If we include 1, 1/1 = 1 > 1/13. So 1 cannot be in the subset if x > 1.
        # If we include 2, 1/2 = 0.5. If x=3, 1/3 < 1/2. So 2 cannot be in subset for x>=3.
        # Generally, if 1/y > 1/x, then y < x.
        # So the subset can only contain elements y < x.
        # This significantly reduces the search space!
        
        # Filter others to only those y < x
        candidates = [y for y in others if y < x]
        
        if not candidates:
            continue
            
        # Now check if any subset of candidates sums to 1/x
        # candidates size is at most x-2 (since y < x and y != x, and y >= 1)
        # For x=24, candidates can be up to 22 elements.
        # 2^22 is ~ 4M, which might be too slow if done for every x in a set of size 15.
        # But we can use DP or meet-in-the-middle.
        # Or simply, since candidates are small, we can iterate subsets.
        # Let's use a simple DP or set of sums.
        
        possible_sums = {Fraction(0, 1)}
        for y in candidates:
            new_sums = set()
            for s in possible_sums:
                ns = s + Fraction(1, y)
                if ns <= target: # Prune sums that exceed target
                    new_sums.add(ns)
            possible_sums.update(new_sums)
            if target in possible_sums:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Initial best
    best = list(range(13, 25))
    if not _check_valid(best):
        best = [1]
    
    # Strategy 1: Greedy from various orderings
    orderings = [
        list(range(24, 0, -1)),  # large to small
        list(range(1, 25)),      # small to large
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19, 23], # Geometric-ish
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        current = []
        for x in ordering:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if _check_valid(candidate):
                current = candidate
        if len(current) > len(best):
            best = current
    
    # Strategy 2: Local Search (Remove 1, Add 2)
    # We try to improve the current best by swapping elements.
    improved = list(best)
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(improved)):
            if time.time() > deadline - 5:
                break
            for j in range(1, 25):
                if time.time() > deadline - 5:
                    break
                if j in improved:
                    continue
                for k in range(j + 1, 25):
                    if time.time() > deadline - 5:
                        break
                    if k in improved:
                        continue
                    candidate = [x for idx, x in enumerate(improved) if idx != i] + [j, k]
                    if _check_valid(candidate):
                        if len(candidate) > len(improved):
                            improved = candidate
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
        
        if len(improved) > len(best):
            best = improved
    
    # Strategy 3: Try adding single elements
    if time.time() < deadline - 10:
        current = list(best)
        for x in range(1, 25):
            if time.time() > deadline - 10:
                break
            if x not in current:
                candidate = current + [x]
                if _check_valid(candidate):
                    current = candidate
        if len(current) > len(best):
            best = current

    return best
