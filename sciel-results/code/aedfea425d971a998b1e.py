import time
from itertools import combinations
from math import lcm
from functools import reduce

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated_incremental(current_sums, new_val):
    """
    Check if adding new_val to a set with subset sums 'current_sums' 
    results in a dissociated set.
    Returns (True, new_sums) if valid, (False, None) otherwise.
    """
    # The new sums will be current_sums U {s + new_val for s in current_sums}
    # We need to ensure no collisions within the new part, and between new part and old part.
    # Since current_sums is a set of unique sums, s1 + new_val == s2 + new_val implies s1 == s2, so no internal collisions in the new part.
    # We only need to check if any s + new_val is in current_sums.
    new_sums = set()
    for s in current_sums:
        ns = s + new_val
        if ns in current_sums or ns in new_sums:
            return False, None
        new_sums.add(ns)
    
    # Combine
    full_sums = current_sums.union(new_sums)
    return True, full_sums

def _find_max_dissociated_subset():
    # Values corresponding to 1/x multiplied by LCM
    # val[x] = L // x
    vals = {x: _L // x for x in range(1, 23)}
    
    # We want to find the largest subset of keys from 1..22
    # Let's try to find the maximum cardinality k
    # We can use a backtracking search.
    # To maximize the chance of finding a large set, we should probably pick elements that are "independent".
    
    # Let's iterate k from 22 down to 1
    # For each k, we try to find a subset of size k.
    
    # A simple heuristic: Sort candidates. 
    # Which order? 
    # Let's try random shuffles or specific orders.
    # Given the small N, we can try a greedy approach first to get a baseline, then backtrack.
    
    # Let's use a DFS with pruning.
    # State: (index, current_subset, current_sums)
    # But we want to maximize size, so we should try to include as many as possible.
    
    # Alternative: Since N=22 is small, we can just try all combinations of size k?
    # C(22, 11) ~ 700k. C(22, 16) ~ 74k.
    # Checking each takes O(2^k).
    # For k=16, 74k * 2^16 ~ 5e9 operations. Too slow in Python.
    
    # So we need the incremental check.
    # The incremental check for a fixed path of length k takes O(2^k).
    # If we do a DFS, the total work depends on the tree size.
    
    # Let's try a randomized greedy approach with restarts.
    # 1. Shuffle 1..22
    # 2. Iterate through shuffled list, add element if it keeps the set dissociated.
    # 3. Record the size.
    # 4. Repeat many times.
    # 5. Try to improve the best found set by swapping?
    
    # Or, let's try to find the maximum k by checking if there exists a dissociated set of size k.
    # We can use a SAT-like solver or just backtracking.
    
    # Let's start with a strong heuristic:
    # The set of powers of 2 is dissociated. 
    # 1, 2, 4, 8, 16 are in 1..22.
    # What about others?
    
    # Let's implement a backtracking search that tries to build a set of size k.
    # We will try k from 22 down to 1.
    # To speed up, we can use the fact that if we can't find a set of size k, we reduce k.
    
    # Optimization: 
    # If we are at depth d in the backtracking for target size k, 
    # and we have (k - d) elements left to pick from (22 - d) candidates,
    # we can prune if (22 - d) < (k - d).
    
    def backtrack(start_idx, current_sums, current_size, target_size, current_subset):
        if current_size == target_size:
            return current_subset[:]
        
        # Pruning: not enough elements left
        remaining_elements = 22 - start_idx
        needed = target_size - current_size
        if remaining_elements < needed:
            return None
        
        # Try to include each element from start_idx to 21
        for i in range(start_idx, 22):
            x = i + 1 # 1-based index
            v = vals[x]
            valid, new_sums = _is_dissociated_incremental(current_sums, v)
            if valid:
                current_subset.append(x)
                result = backtrack(i + 1, new_sums, current_size + 1, target_size, current_subset)
                if result is not None:
                    return result
                current_subset.pop()
            
            # Optional: If we don't pick i, can we still reach target_size?
            # Yes, handled by the loop continuing.
            
        return None

    # We need to be careful with the order of elements.
    # If we just iterate 1..22, we might miss good combinations.
    # However, for finding *any* solution of size k, the order doesn't change the existence, 
    # but it affects the speed.
    # To maximize speed, we might want to try "promising" elements first.
    # But since we want to find the *max* k, and we try k from high to low,
    # if a solution exists, backtracking will find it.
    
    # The worst case is exponential.
    # For k=22, it will fail quickly (likely at the first few steps).
    # For k=16, it might take longer.
    
    # Let's add a time limit guard.
    start_time = time.time()
    deadline = start_time + 80.0 # 10s buffer
    
    # Try k from 22 down to 1
    for k in range(22, 0, -1):
        if time.time() > deadline:
            break
        # To help the search, we can shuffle the order of candidates?
        # No, backtracking relies on the fixed order for determinism and pruning logic (start_idx).
        # But we can permute the global list of candidates?
        # If we permute the candidates, we change the problem instance order.
        # Let's try a few random permutations if the first one fails?
        # Actually, for k=22, 21, etc., it will likely fail fast.
        # For smaller k, it might succeed.
        
        # Let's just run the backtracking on the natural order 1..22.
        # If it takes too long, we might need to optimize.
        # Given N=22, 2^22 is 4M. The backtracking tree is much smaller if pruning works.
        
        result = backtrack(0, {0}, 0, k, [])
        if result is not None:
            return result
            
    # Fallback: Return a known good set or the best found so far?
    # The seed returned size 9.
    # We should return the best valid set we can find quickly.
    # Let's do a quick greedy search if the above fails or times out.
    
    def greedy_construct():
        used = set()
        current_sums = {0}
        result = []
        # Sort by some heuristic? 
        # Let's try random shuffles for a few iterations
        import random
        best_result = []
        for _ in range(100):
            candidates = list(range(1, 23))
            random.shuffle(candidates)
            cur_sums = {0}
            cur_set = []
            for x in candidates:
                v = vals[x]
                valid, new_sums = _is_dissociated_incremental(cur_sums, v)
                if valid:
                    cur_sums = new_sums
                    cur_set.append(x)
            if len(cur_set) > len(best_result):
                best_result = cur_set[:]
        return best_result

    return greedy_construct()

def construct():
    return _find_max_dissociated_subset()
