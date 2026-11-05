"""Improved program: maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Try multiple initialization strategies
    candidates = []
    
    # Strategy 1: Start with {67,...,100}
    S1 = set(range(67, 101))
    for x in range(1, 67):
        if time.time() - _START > _DEADLINE - 15.0:
            break
        if not _has_subset_sum_200_fast(S1 | {x}):
            S1.add(x)
    candidates.append(S1)
    
    # Strategy 2: Start with {66,...,100}
    S2 = set(range(66, 101))
    for x in range(1, 66):
        if time.time() - _START > _DEADLINE - 15.0:
            break
        if not _has_subset_sum_200_fast(S2 | {x}):
            S2.add(x)
    candidates.append(S2)
    
    # Strategy 3: Greedy from 100 down to 1
    S3 = set()
    for x in range(100, 0, -1):
        if time.time() - _START > _DEADLINE - 15.0:
            break
        if not _has_subset_sum_200_fast(S3 | {x}):
            S3.add(x)
    candidates.append(S3)
    
    # Strategy 4: Greedy from 1 up to 100
    S4 = set()
    for x in range(1, 101):
        if time.time() - _START > _DEADLINE - 15.0:
            break
        if not _has_subset_sum_200_fast(S4 | {x}):
            S4.add(x)
    candidates.append(S4)
    
    # Pick the largest candidate
    S = max(candidates, key=len)
    
    # Local search
    improved = True
    while improved and time.time() - _START < _DEADLINE - 10.0:
        improved = False
        S_list = sorted(S)
        
        # Phase 1: Try pure additions
        for j in range(1, 101):
            if j in S:
                continue
            if time.time() - _START > _DEADLINE - 10.0:
                break
            if not _has_subset_sum_200_fast(S | {j}):
                S.add(j)
                improved = True
        
        # Phase 2: Try 1-for-1 swaps to enable future additions
        S_list = sorted(S)
        for i in range(len(S_list)):
            if time.time() - _START > _DEADLINE - 10.0:
                break
            for j in range(1, 101):
                if j in S:
                    continue
                new_S = S - {S_list[i]} | {j}
                if not _has_subset_sum_200_fast(new_S):
                    # Try this swap and see if it enables additions
                    S = new_S
                    improved = True
                    break
            if improved:
                break
        
        # Phase 3: Try 1-for-2 swaps (increase size by 1)
        S_list = sorted(S)
        for i in range(len(S_list)):
            if time.time() - _START > _DEADLINE - 10.0:
                break
            for j1 in range(1, 51):
                if j1 in S:
                    continue
                for j2 in range(j1 + 1, 101):
                    if j2 in S:
                        continue
                    if time.time() - _START > _DEADLINE - 10.0:
                        break
                    new_S = S - {S_list[i]} | {j1, j2}
                    if len(new_S) > len(S) and not _has_subset_sum_200_fast(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(S)

def _has_subset_sum_200_fast(elements):
    """Check if any subset of elements sums to 200 using bitset."""
    target = 200
    bits = 1  # bit 0 is set
    
    for x in elements:
        if x > target:
            continue
        bits |= (bits << x)
        if (bits >> target) & 1:
            return True
    
    return False

def _has_subset_sum_200(elements):
    """Check if any subset of elements sums to 200."""
    target = 200
    dp = [False] * (target + 1)
    dp[0] = True
    
    for x in elements:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if dp[s]:
                dp[s + x] = True
                if dp[target]:
                    return True
    
    return dp[target]

def _find_subset_sum_200(elements):
    """Find one subset of elements that sums to 200. Returns list of elements or None."""
    elements = sorted(elements, reverse=True)
    target = 200
    parent = [None] * (target + 1)
    parent[0] = (None, None)
    
    for x in elements:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if parent[s] is not None and parent[s + x] is None:
                parent[s + x] = (s, x)
        
        if parent[target] is not None:
            subset = []
            s = target
            while s > 0:
                prev_s, x = parent[s]
                subset.append(x)
                s = prev_s
            return subset
    
    return None

def _count_subsets_sum_200(elements):
    """Count the number of subsets of elements that sum to 200."""
    target = 200
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for x in elements:
        if x > target:
            continue
        for s in range(target - x, -1, -1):
            if dp[s] > 0:
                dp[s + x] += dp[s]
                
    return dp[target]

def _greedy_fallback():
    """Original greedy approach as fallback."""
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
