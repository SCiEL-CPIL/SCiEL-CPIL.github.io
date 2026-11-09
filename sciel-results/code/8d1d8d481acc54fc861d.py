"""Search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import sys
from math import gcd
from functools import reduce

# LCM of 1..22
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 23))
# LCM = 232792560

# Weights: w_i = LCM // i for i in 1..22
_WEIGHTS = [None] + [_LCM // i for i in range(1, 23)]

def _verify(subset):
    """Check if all 2^k subset sums of weights are distinct."""
    k = len(subset)
    sums = {0}
    for i in subset:
        w = _WEIGHTS[i]
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin
    
    # Try from high cardinality down
    # Pre-sort indices by weight ascending for better pruning
    indices = list(range(1, 23))
    indices.sort(key=lambda i: _WEIGHTS[i])
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback
    
    # For each target size from 22 down to 10
    for target_size in range(22, 9, -1):
        if time.time() > deadline:
            break
        # DFS to find a subset of size target_size
        found = _dfs_find(indices, target_size, start_time, deadline)
        if found is not None:
            best = found
            break
    
    return best

def _dfs_find(all_indices, target_size, start_time, deadline):
    """Find a subset of size target_size with all subset sums distinct."""
    n = len(all_indices)
    # We'll use backtracking
    # State: current list of selected indices, current set of subset sums
    
    def dfs(start_idx, current, sums):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            return list(current)
        
        remaining = target_size - len(current)
        # Pruning: not enough elements left
        if n - start_idx < remaining:
            return None
        
        for i in range(start_idx, n):
            idx = all_indices[i]
            w = _WEIGHTS[idx]
            
            # Check if adding w causes collision
            # New sums would be sums ∪ {s + w for s in sums}
            # Collision if any (s1 + w) == s2 for s1, s2 in sums, or (s1+w) == (s2+w)
            # Since all s in sums are distinct, (s1+w)==(s2+w) iff s1==s2, so no issue there.
            # Collision iff (s + w) in sums for some s in sums
            
            # Efficient check: for each s in sums, if (s+w) in sums, collision
            # But sums can be large. Alternative: check if w can be expressed as difference of two sums
            # i.e., exists s1, s2 in sums with s1 - s2 = w (or s2 - s1 = w)
            
            # Actually simpler: new_sums = {s + w for s in sums}. Collision iff new_sums & sums != empty
            # Or equivalently, exists s in sums such that (s+w) in sums
            
            # For efficiency, convert sums to a set (it already is)
            has_collision = False
            for s in sums:
                if (s + w) in sums:
                    has_collision = True
                    break
            
            if not has_collision:
                new_sums = sums | {s + w for s in sums}
                current.append(idx)
                result = dfs(i + 1, current, new_sums)
                current.pop()
                if result is not None:
                    return result
        
        return None
    
    return dfs(0, [], {0})
