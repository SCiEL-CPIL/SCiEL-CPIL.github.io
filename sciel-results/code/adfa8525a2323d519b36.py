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
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Indices sorted by weight descending (largest weights first)
    # This is generally better for dissociated sets as it creates larger gaps
    indices = list(range(1, 23))
    indices.sort(key=lambda i: _WEIGHTS[i], reverse=True)
    
    # Fallback: A known good set if search fails or times out
    # {1, 2, 3, 5, 7, 11, 13, 17, 19} has size 9
    # Let's try to find a larger one.
    # Based on previous runs, size 16 was found.
    # Let's initialize best with a set of size 16 if we can construct one quickly,
    # or just start the search from 22 down.
    
    best = None
    
    # Try to find a subset of size target_size
    for target_size in range(22, 0, -1):
        if time.time() > deadline:
            break
        
        # If we found a solution of this size, we can return it immediately
        # because we are iterating from largest to smallest.
        # However, we should verify it just in case.
        
        found = _dfs_find(indices, target_size, start_time, deadline)
        if found is not None:
            # Verify the found solution
            if _verify(found):
                return found
            else:
                # Should not happen if _dfs_find is correct, but safety check
                continue
        
        # If we didn't find one, try smaller size
        # Note: _dfs_find might take a long time. 
        # If it times out, it returns None.
        
    # If loop finishes without finding anything (shouldn't happen as size 1 always works)
    # Return a valid small set
    return [1]

def _dfs_find(all_indices, target_size, start_time, deadline):
    """Find a subset of size target_size with all subset sums distinct."""
    n = len(all_indices)
    
    # We use a recursive helper
    def dfs(start_idx, current, sums):
        # Time check
        if time.time() > deadline:
            return None
        
        # If we have enough elements, we are done
        if len(current) == target_size:
            return list(current)
        
        # Pruning: not enough elements left
        remaining_needed = target_size - len(current)
        if n - start_idx < remaining_needed:
            return None
        
        # Try adding each candidate from start_idx onwards
        for i in range(start_idx, n):
            idx = all_indices[i]
            w = _WEIGHTS[idx]
            
            # Check for collision
            # Collision if (s + w) is already in sums for any s in sums
            has_collision = False
            # Optimization: if w is very large, it's less likely to collide
            # But we must check.
            for s in sums:
                if (s + w) in sums:
                    has_collision = True
                    break
            
            if not has_collision:
                # Create new sums set
                # sums is a set of ints
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w)
                
                current.append(idx)
                result = dfs(i + 1, current, new_sums)
                current.pop()
                
                if result is not None:
                    return result
        
        return None
    
    return dfs(0, [], {0})
