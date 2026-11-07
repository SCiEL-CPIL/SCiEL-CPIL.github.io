"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import sys
import time
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    We use a backtracking search with optimized incremental validity checks.
    Candidates are tried in descending order to prioritize large numbers,
    which reduces the likelihood of divisibility conflicts.
    """
    
    start_time = time.time()
    timeout = 58  # Leave 2s for overhead
    
    best_set = []
    
    # Precompute divisors for all possible subset sums?
    # Max sum: 30 * 256 = 7680.
    # Min sum: 1.
    max_possible_sum = 30 * 256
    divisors = [[] for _ in range(max_possible_sum + 1)]
    for i in range(1, max_possible_sum + 1):
        for j in range(i, max_possible_sum + 1, i):
            divisors[j].append(i)
            
    # Also precompute multiples? Or just check on fly.
    # Multiples check: for a value v, check if k*v is in S.
    
    def check_addition(current_sums_list, current_sums_set, x):
        """
        Check if adding x to the set represented by current_sums_list/set is valid.
        Returns (new_sums_list, new_sums_set) if valid, else None.
        """
        # Generate new sums: x and s+x for all s in current_sums_list
        # Note: current_sums_list contains sums of non-empty subsets of the current set.
        # The new element x forms subsets: {x} and {x} U T for T in current subsets.
        # So new sums are x and {s + x for s in current_sums_list}.
        
        new_sums = [x] + [s + x for s in current_sums_list]
        
        # 1. Check for distinctness
        # If any new sum is in current_sums_set, invalid.
        # If any two new sums are equal, invalid.
        if len(set(new_sums)) != len(new_sums):
            return None
            
        new_sums_set = set(new_sums)
        if new_sums_set & current_sums_set:
            return None
        
        # 2. Check divisibility
        # A) new_sum divides current_sum
        # B) current_sum divides new_sum
        # C) new_sum1 divides new_sum2
        
        for ns in new_sums:
            # Check if ns divides any cs in current_sums_set
            # Iterate multiples of ns
            if ns > 0:
                m = ns * 2
                while m <= max_possible_sum:
                    if m in current_sums_set:
                        return None
                    m += ns
            
            # Check if any cs in current_sums_set divides ns
            # Iterate divisors of ns
            for d in divisors[ns]:
                if d != ns and d in current_sums_set:
                    return None
            
        # Check divisibility among new_sums
        # Sort to check i < j
        new_sums_sorted = sorted(new_sums)
        for i in range(len(new_sums_sorted)):
            ns_i = new_sums_sorted[i]
            # Check if ns_i divides any later new sum
            # We can just check multiples
            m = ns_i * 2
            while m <= max_possible_sum:
                if m in new_sums_set:
                    return None
                m += ns_i
            
            # Check if any earlier new sum divides ns_i
            # Since sorted, earlier sums are smaller.
            # We already checked divisors for "current", but now we check "new".
            # Actually, the loop above "ns divides any cs" and "cs divides ns" covers pairs (new, current).
            # For (new, new), we need to check both directions.
            # The "multiples" check above handles ns_i | ns_j for i < j.
            # We also need to check if any ns_k (k < i) divides ns_i.
            # This is equivalent to checking divisors of ns_i and seeing if they are in new_sums_set.
            for d in divisors[ns_i]:
                if d != ns_i and d in new_sums_set:
                    return None

        return new_sums, new_sums_set

    def backtrack(current_set, current_sums_list, current_sums_set, start_idx):
        nonlocal best_set, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if len(current_set) > len(best_set):
            best_set = list(current_set)
            
        # If we've tried all candidates, return
        if start_idx >= len(candidates):
            return
            
        # Try adding candidates from start_idx to end
        # We iterate in reverse order (largest to smallest)
        for i in range(len(candidates) - 1, start_idx - 1, -1):
            if time.time() - start_time > timeout:
                return
                
            x = candidates[i]
            
            # Pruning: if x is too small, might not help? No, just try.
            
            result = check_addition(current_sums_list, current_sums_set, x)
            if result:
                new_sums_list, new_sums_set = result
                new_set = current_set + [x]
                
                # Recurse
                # Next candidates must be smaller than x to maintain some order?
                # Actually, we just pick from remaining candidates.
                # To avoid permutations, we enforce increasing index in the original list?
                # No, we are iterating i downwards.
                # The next recursive call should look at candidates with index < i.
                backtrack(new_set, new_sums_list, new_sums_set, i - 1)
                
                # Backtrack
                # No state to undo explicitly as we pass copies (lists/sets are immutable in this context effectively, or we create new ones)
                # Actually, we create new lists/sets in check_addition, so no mutation.

    # Candidates: 256 down to 1
    candidates = list(range(256, 0, -1))
    
    # Initial state: empty set
    # Sums of empty set: none.
    # But our check_addition expects current_sums_list to be sums of non-empty subsets.
    # For the first element, current_sums_list is empty.
    # new_sums = [x] + [] = [x].
    # This works.
    
    backtrack([], [], set(), 256)
    
    if not best_set:
        # Fallback to a known valid small set if search fails (shouldn't happen)
        best_set = [31, 32, 34]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
