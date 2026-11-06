"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))
    
    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]

    # We want to maximize the size of the subset.
    # The maximum possible size is 22.
    # We will try to find a valid subset of size 22, then 21, etc.
    
    # To speed up the search, we can use a heuristic ordering.
    # Let's try sorting by reciprocal value.
    # Indices 0..21 correspond to values 1..22.
    # recip[0] = L, recip[1] = L/2, etc.
    
    # Strategy: 
    # 1. Check if the full set {1..22} is valid.
    # 2. If not, use backtracking to find the largest valid subset.
    
    def is_dissociated(indices):
        """Check if the set of indices is dissociated."""
        sums = {0}
        for idx in indices:
            r = recip[idx]
            new_sums = {s + r for s in sums}
            # Check for collision
            if len(new_sums | sums) != len(sums) * 2:
                return False
            sums = new_sums | sums
        return True

    # First, quick check for the full set
    full_set = list(range(22))
    if is_dissociated(full_set):
        return [i + 1 for i in full_set]

    # If full set is not valid, we need to find the max cardinality subset.
    # We use backtracking.
    # We try to build a set of size K, starting from K=21 down to 1.
    # We use a heuristic to order the candidates.
    
    # Heuristic: Sort indices by reciprocal value.
    # Let's try adding elements with larger reciprocals (smaller indices) first?
    # Or smaller reciprocals (larger indices) first?
    # Let's try a few orders.
    
    def backtrack(selected, remaining, sums, target_size):
        """
        selected: list of indices selected so far
        remaining: list of indices available to select
        sums: set of current subset sums (integers)
        target_size: desired size of the final set
        """
        nonlocal best_result, best_size
        
        if time.time() > deadline:
            return False
            
        current_size = len(selected)
        
        if current_size == target_size:
            if target_size > best_size:
                best_result = selected[:]
                best_size = target_size
            return True
            
        # Pruning: not enough remaining elements
        if current_size + len(remaining) < target_size:
            return False
            
        # Pruning: even if we take all remaining, we can't beat best_size?
        # This is handled by the outer loop trying decreasing target_size.
        
        for i in range(len(remaining)):
            idx = remaining[i]
            r = recip[idx]
            
            # Check collision
            # new_sums = {s + r for s in sums}
            # Collision if any s + r is in sums, or any s + r == t + r (impossible if sums distinct)
            # So collision if (sums + r) intersects sums.
            
            # Efficient check:
            # If we add r, the new sums are sums U (sums + r).
            # They are distinct iff sums and (sums + r) are disjoint.
            # i.e., (sums + r) & sums is empty.
            
            # Constructing the set is O(N).
            # We can do:
            # for s in sums:
            #     if s + r in sums: return collision
            
            # This is O(N) on average for set lookup.
            # Let's try this to avoid creating a new set if there's a collision.
            
            has_collision = False
            for s in sums:
                if s + r in sums:
                    has_collision = True
                    break
            
            if has_collision:
                continue
                
            # No collision, proceed
            selected.append(idx)
            # Update sums
            new_sums = sums | {s + r for s in sums}
            
            if backtrack(selected, remaining[i+1:], new_sums, target_size):
                selected.pop()
                return True
                
            selected.pop()
            
        return False

    best_result = [1] # Fallback
    best_size = 1
    
    # Try to find sets of size 21, 20, ...
    # We can use a specific ordering for 'remaining'.
    # Let's try two orderings: ascending index and descending index.
    
    orderings = [
        list(range(22)),
        list(range(21, -1, -1)),
        sorted(range(22), key=lambda x: recip[x]),
        sorted(range(22), key=lambda x: -recip[x])
    ]
    
    found = False
    for target in range(21, 0, -1):
        if time.time() > deadline:
            break
            
        for order in orderings:
            if time.time() > deadline:
                break
                
            # We need to search for a subset of size 'target' from 'order'.
            # The backtracking function assumes 'remaining' is the list of candidates.
            # We should pass 'order' as the initial remaining list.
            
            # Optimization: If we are looking for a specific target, 
            # we can prune branches that cannot reach it.
            
            if backtrack([], order, {0}, target):
                found = True
                break
                
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
