"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import sys
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct (no two different subsets have the same sum).
    2. No two distinct subset sums divide one another.
    
    Strategy:
    - Use a greedy/backtracking approach with incremental validity checking.
    - Precompute subset sums incrementally as we add elements.
    - Use divisibility checks efficiently by maintaining sorted subset sums.
    - Try to maximize set size within the 60-second time limit.
    """
    
    def check_divisibility(sums_set):
        """Check if any two distinct sums in sums_set divide each other."""
        sorted_sums = sorted(sums_set)
        for i in range(len(sorted_sums)):
            s1 = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                s2 = sorted_sums[j]
                if s2 % s1 == 0:
                    return False
        return True
    
    def get_subset_sums(elements):
        """Generate all nonempty subset sums for a list of elements."""
        if not elements:
            return set()
        sums = set()
        for mask in range(1, 1 << len(elements)):
            s = 0
            for i in range(len(elements)):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums
    
    def is_valid(elements):
        """Check if a set of elements satisfies both constraints."""
        if not elements:
            return True
        # First check: all subset sums must be distinct
        # This is equivalent to: no two different subsets have the same sum
        # For a set of n elements, there are 2^n - 1 nonempty subsets
        # We need all 2^n - 1 sums to be distinct
        sums = get_subset_sums(elements)
        if len(sums) != (1 << len(elements)) - 1:
            return False
        # Second check: no two distinct sums divide each other
        return check_divisibility(sums)
    
    # Greedy construction with backtracking
    def find_max_set(max_size=15, timeout=55):
        start_time = time.time()
        best = []
        best_size = 0
        
        # Try elements in descending order (larger elements tend to create larger gaps)
        # This helps avoid small numbers that create many divisibility conflicts
        candidates = list(range(256, 0, -1))
        
        def backtrack(current_set, start_idx):
            nonlocal best, best_size, start_time
            if time.time() - start_time > timeout:
                return
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best = list(current_set)
            
            if len(current_set) >= max_size:
                return
            
            # Pruning: if we can't beat best_size, return
            if len(current_set) + (len(candidates) - start_idx) <= best_size:
                return
            
            for idx in range(start_idx, len(candidates)):
                cand = candidates[idx]
                new_set = current_set + [cand]
                
                # Quick check: verify validity of new_set
                if is_valid(new_set):
                    backtrack(new_set, idx + 1)
                
                # Early termination if time is up
                if time.time() - start_time > timeout:
                    return
        
        backtrack([], 0)
        return best
    
    # Also try a different ordering: ascending might find different solutions
    def find_max_set_ascending(max_size=15, timeout=55):
        start_time = time.time()
        best = []
        best_size = 0
        
        candidates = list(range(1, 257))
        
        def backtrack(current_set, start_idx):
            nonlocal best, best_size, start_time
            if time.time() - start_time > timeout:
                return
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best = list(current_set)
            
            if len(current_set) >= max_size:
                return
            
            for idx in range(start_idx, len(candidates)):
                cand = candidates[idx]
                new_set = current_set + [cand]
                
                if is_valid(new_set):
                    backtrack(new_set, idx + 1)
                
                if time.time() - start_time > timeout:
                    return
        
        backtrack([], 0)
        return best
    
    # Try both orderings and take the best result
    result1 = find_max_set(max_size=12, timeout=25)
    result2 = find_max_set_ascending(max_size=12, timeout=25)
    
    best_result = result1 if len(result1) >= len(result2) else result2
    
    # If we didn't find anything, fall back to a known small valid set
    if not best_result:
        # Verify [3, 5] is valid: sums are 3, 5, 8. 3|5? No. 3|8? No. 5|8? No. Valid.
        best_result = [3, 5]
    
    # Ensure we return Python ints in a list
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
