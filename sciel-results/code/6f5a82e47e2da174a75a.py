import time
from math import lcm
import bisect
import random

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# Precompute LCM of 1..24
_L = 1
for i in range(1, 25):
    _L = lcm(_L, i)

# Precompute integer reciprocals: R[x] = _L // x
_R = [0] * 25
for i in range(1, 25):
    _R[i] = _L // i

def _subset_sum_exists(target, elements):
    """
    Check if target can be formed by a non-empty subset of elements.
    elements is a list of integers (scaled reciprocals).
    target is an integer (scaled reciprocal).
    """
    if not elements or target <= 0:
        return False
    
    n = len(elements)
    
    if n <= 15:
        reachable = {0}
        for e in elements:
            new_reachable = set()
            for s in reachable:
                if s <= target:
                    new_reachable.add(s)
                if s + e <= target:
                    new_reachable.add(s + e)
            reachable = new_reachable
        return target in reachable
    
    # Meet in the middle for larger n
    mid = n // 2
    left = elements[:mid]
    right = elements[mid:]
    
    sums_left = [0]
    for e in left:
        new_sums = [s + e for s in sums_left if s + e <= target]
        sums_left.extend(new_sums)
        
    sums_right = [0]
    for e in right:
        new_sums = [s + e for s in sums_right if s + e <= target]
        sums_right.extend(new_sums)
        
    sums_right.sort()
    
    for s_l in sums_left:
        if s_l > target:
            continue
        needed = target - s_l
        idx = bisect.bisect_left(sums_right, needed)
        if idx < len(sums_right) and sums_right[idx] == needed:
            return True
            
    return False

def _can_add(k, S):
    """
    Check if adding element k to set S results in a valid set.
    S is a list of integers (elements, not reciprocals).
    k is an integer not in S.
    """
    R_k = _R[k]
    R_S = [_R[x] for x in S]
    
    # Check 1: Is R_k a subset sum of R_S?
    if _subset_sum_exists(R_k, R_S):
        return False
        
    # Check 2: For each y in S, does adding k create a violation for y?
    # Violation for y: R_y == sum(subset of R_S \ {R_y} + {R_k})
    # This implies R_y - R_k == sum(subset of R_S \ {R_y})
    for y in S:
        R_y = _R[y]
        if R_y <= R_k:
            continue # R_y - R_k <= 0, cannot be sum of positive reciprocals
            
        others = [_R[x] for x in S if x != y]
        target = R_y - R_k
        if _subset_sum_exists(target, others):
            return False
            
    return True

def _greedy_with_order(order):
    """Greedy construction using a specific element order."""
    S = []
    for k in order:
        if _can_add(k, S):
            S.append(k)
    return S

def construct():
    # Initial greedy to find a good lower bound
    orders = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
    ]
    
    best_set = []
    for order in orders:
        if _time_remaining() < 5:
            break
        S_g = _greedy_with_order(order)
        if len(S_g) > len(best_set):
            best_set = S_g
            
    best_size = len(best_set)
    
    # Branch and Bound
    # We process elements from 24 down to 1.
    # State: idx (current element to consider, starting at 24), current_set, current_size
    # We use recursion.
    
    def dfs(idx, current_set, current_size):
        nonlocal best_set, best_size
        
        if _time_remaining() < 1:
            return
            
        # Pruning: even if we add all remaining elements, we can't beat best_size
        if current_size + (idx - 1 + 1) <= best_size: # Elements 1..idx are remaining? No, idx is the current one.
             # We are considering element 'idx'. Remaining are 1..idx.
             # Max additional is idx.
             if current_size + idx <= best_size:
                 return
        
        # Try to include idx
        if _can_add(idx, current_set):
            new_set = current_set + [idx]
            if len(new_set) > best_size:
                best_size = len(new_set)
                best_set = new_set[:]
                
            # Recurse on idx-1
            dfs(idx - 1, new_set, len(new_set))
            
        # Try to exclude idx
        dfs(idx - 1, current_set, current_size)

    if _time_remaining() > 5:
        # Start DFS from 24 down to 1
        # Note: The order in DFS matters for finding large sets quickly.
        # Descending order (24, 23, ...) is good because large numbers are "safe".
        dfs(24, [], 0)
        
    if not best_set:
        return list(range(13, 25))
        
    return sorted(best_set)

if __name__ == "__main__":
    s = construct()
    print(s)
    print(len(s))
