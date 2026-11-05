"""Improved program: maximum-cardinality subset of [1,100] with no subset summing to 200."""
import time

_DEADLINE = 90.0
_START = None

def construct():
    global _START
    if _START is None:
        _START = time.time()
    
    # Start with known-good set {67,...,100} = 34 elements
    S = set(range(67, 101))
    
    # Try to add elements from 1 to 66
    for x in range(1, 67):
        if time.time() - _START > _DEADLINE - 5.0:
            break
        if _has_subset_sum_200(S | {x}):
            continue
        S.add(x)
    
    # Local search: try swapping elements to increase size
    improved = True
    while improved and time.time() - _START < _DEADLINE - 5.0:
        improved = False
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(1, 101):
                if j in S_list:
                    continue
                if time.time() - _START > _DEADLINE - 5.0:
                    break
                new_S = S - {S_list[i]} | {j}
                if len(new_S) > len(S) and not _has_subset_sum_200(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
        # Also try pure additions if we haven't filled
        for j in range(1, 101):
            if j in S:
                continue
            if time.time() - _START > _DEADLINE - 5.0:
                break
            if not _has_subset_sum_200(S | {j}):
                S.add(j)
                improved = True
    
    return list(S)

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
