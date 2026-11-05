"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 22
    candidates = list(range(1, N + 1))
    
    # We search for the maximum size K
    # We try K from 11 down to 1
    # If we find a valid set of size K, we return it.
    
    # Helper to check if a set is dissociated using integer arithmetic
    # State: list of elements, set of scaled sums, current LCM
    def is_dissociated_int(elements):
        if not elements:
            return True
        L = 1
        for x in elements:
            L = L * x // gcd(L, x)
        
        # Scaled sums: sum(1/x for x in subset) * L
        # Initialize with 0 (empty subset)
        sums = {0}
        
        for x in elements:
            new_sums = set()
            scale = L // x
            for s in sums:
                new_sums.add(s + scale)
            # Check for collision: new_sums should not intersect with existing sums?
            # No, the total set of sums is sums U (sums + new_sums).
            # We need all sums in (sums U new_sums) to be distinct.
            # This is equivalent to sums being distinct, new_sums being distinct (guaranteed if sums are),
            # and sums being disjoint from new_sums.
            if sums & new_sums:
                return False
            sums = sums | new_sums
            
        return True

    # Backtracking search for a set of size K
    def find_set_of_size(K, candidates):
        # We want to find a subset of candidates of size K that is dissociated
        # To make search faster, we can prune.
        # Order candidates: smaller numbers (larger reciprocals) might be harder to fit together?
        # Or easier?
        # Let's try sorting by value descending (22, 21, ...) or ascending (1, 2, ...)?
        # Usually, including 1 is very restrictive.
        # Let's try to exclude 1 first? Or include it last?
        
        # Let's use a recursive backtracker
        # current_set: list of chosen elements
        # current_sums: set of scaled sums? 
        # Maintaining scaled sums dynamically is tricky because LCM changes.
        # Alternative: Just use the integer check at the leaf? 
        # No, that's too slow. We need pruning.
        
        # Let's stick to the float check for speed, but verify with integers at the end?
        # Or use a robust float check with tolerance?
        # Given N=22, the number of subsets is small.
        # Let's try a simpler backtracking with float sums.
        
        def backtrack(start_idx, current_set, current_sums):
            # current_sums is a set of floats
            if time.time() - start_time > timeout - 5:
                return None
            
            if len(current_set) == K:
                # Verify with integers to be safe
                if is_dissociated_int(current_set):
                    return current_set
                return None
            
            # Pruning: not enough candidates left
            if len(candidates) - start_idx < K - len(current_set):
                return None
                
            # Try adding each candidate from start_idx onwards
            for i in range(start_idx, len(candidates)):
                x = candidates[i]
                recip = 1.0 / x
                
                # Check if adding x causes collision
                # New sums are {s + recip for s in current_sums}
                # Collision if any (s + recip) is in current_sums
                # i.e., if there exist s1, s2 in current_sums such that s1 + recip == s2
                # i.e., s2 - s1 == recip
                
                valid = True
                # Optimization: if recip > max(current_sums) - min(current_sums), no collision?
                # Not necessarily, but if recip is larger than the range, no collision.
                # Actually, if current_sums is contiguous-ish, this helps.
                
                # Simple check
                for s in current_sums:
                    if s + recip in current_sums:
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                # Recurse
                new_set = current_set + [x]
                new_sums = current_sums | {s + recip for s in current_sums}
                
                result = backtrack(i + 1, new_set, new_sums)
                if result is not None:
                    return result
                    
            return None

        # Try to find a set of size K
        # We can order candidates to improve pruning.
        # Candidates: 1..22
        # Let's try different orderings or just the natural one.
        # Excluding 1 is often good for dissociated sets.
        
        # Let's try to find any set of size K.
        # We'll iterate through combinations? No, backtracking is better.
        
        # To maximize chances, let's try to pick "small" reciprocals first? 
        # i.e. large x.
        # Let's sort candidates in descending order (22, 21, ..., 1)
        # This might allow packing more elements because they are "small".
        
        sorted_candidates = sorted(candidates, reverse=True)
        
        result = backtrack(0, [], {0.0})
        return result

    # Try sizes from 11 down to 1
    for K in range(11, 0, -1):
        res = find_set_of_size(K, candidates)
        if res is not None:
            return res
            
    # Fallback to a known valid set if search fails or times out
    # The static set from the seed
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

# EVOLVE-BLOCK-END
