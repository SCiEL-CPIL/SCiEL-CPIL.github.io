import time
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _subset_sums_exists(target, values):
    """
    Check if target can be formed as a subset sum of values.
    target and values are integers.
    Returns True if exists, False otherwise.
    """
    if not values:
        return target == 0
    if target == 0:
        return True
    if target < 0:
        return False
    
    n = len(values)
    # Meet in the middle for efficiency if n is large, but n <= 24 so 2^12 is small
    # Actually, simple DP or recursion might be fine, but values can be large.
    # Let's use a set of reachable sums.
    reachable = {0}
    for v in values:
        new_reachable = set()
        for s in reachable:
            new_reachable.add(s + v)
        reachable = reachable.union(new_reachable)
        if target in reachable:
            return True
    return target in reachable

def _find_conflicting_subsets(x, S):
    """
    Find if 1/x is a subset sum of 1/y for y in S, y > x.
    Returns a list of subsets (tuples of y) if found, else empty list.
    """
    larger = [y for y in S if y > x]
    if not larger:
        return []
    
    all_vals = [x] + larger
    lcm_val = reduce(_lcm, all_vals)
    target_int = lcm_val // x
    candidate_vals = [(lcm_val // y, y) for y in larger]
    
    # We need to find subsets of candidate_vals that sum to target_int.
    # Since we only need to know if it exists and maybe which elements, 
    # and the number of larger elements is small, we can enumerate subsets.
    # Max larger elements is 23. 2^23 is too big for full enumeration in Python in a tight loop.
    # But we can prune.
    
    # Better: Use the subset sum check first.
    vals_only = [v for v, _ in candidate_vals]
    if not _subset_sums_exists(target_int, vals_only):
        return []
    
    # If it exists, find one or more minimal subsets to identify conflicts.
    # For the hitting set strategy, we just need to know WHICH y's are involved.
    # Let's find all subsets that sum to target_int.
    # To keep it fast, we'll find a few.
    
    subsets_found = []
    n = len(candidate_vals)
    # Sort by value to help pruning?
    
    # Recursive search for subsets
    def backtrack(idx, current_sum, current_subset):
        if current_sum == target_int:
            subsets_found.append(tuple(current_subset))
            return
        if idx >= n or current_sum > target_int:
            return
        
        # Include candidate_vals[idx]
        val, y = candidate_vals[idx]
        backtrack(idx + 1, current_sum + val, current_subset + [y])
        
        # Exclude candidate_vals[idx]
        backtrack(idx + 1, current_sum, current_subset)
        
    backtrack(0, 0, [])
    
    # We might have many subsets. We just need the elements involved.
    # For the "remove one element" strategy, we can just return the list of all y's involved in ANY subset.
    # Or better, return the list of subsets to allow smarter choice.
    return subsets_found

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initial set
    S = set(range(1, 25))
    
    # Iterative Pruning
    # We repeatedly find invalid elements and remove conflicting elements.
    # To decide which element to remove, we count how many conflicts each element is part of.
    
    while time.time() < deadline:
        invalid_elements = []
        conflict_map = {} # y -> count of conflicts involving y
        
        for x in sorted(S):
            subsets = _find_conflicting_subsets(x, S)
            if subsets:
                invalid_elements.append(x)
                # Mark all y in these subsets as involved in a conflict
                for subset in subsets:
                    for y in subset:
                        conflict_map[y] = conflict_map.get(y, 0) + 1
        
        if not invalid_elements:
            break
        
        # If there are invalid elements, we must remove something.
        # Strategy: Remove the element y that appears in the most conflicts.
        # If tie, remove the largest y (to preserve small numbers which are "harder" to sum? 
        # Actually, small numbers have large reciprocals, so they are "targets". 
        # Large numbers have small reciprocals, they are "sources". 
        # Removing a source (large y) might break many sums.
        # Removing a target (small x) breaks only its own validity but keeps sources.
        # However, if we remove x, we don't need to worry about x anymore.
        # If we remove y, we might still have other invalid x's.
        
        # Let's prioritize removing the y with the highest conflict count.
        # If x itself is in conflict_map? No, x is not in the subsets (subsets contain y > x).
        
        if not conflict_map:
             # Should not happen if invalid_elements is not empty
             break
             
        # Find the y with max count
        max_count = max(conflict_map.values())
        candidates = [y for y, c in conflict_map.items() if c == max_count]
        
        # Tie-breaker: Remove the largest y? Or the smallest?
        # Removing a large y (small reciprocal) is "cheap" in terms of value, but might be critical for sums.
        # Let's remove the largest y to keep the set "dense" with small numbers?
        # Or remove the smallest y to keep large numbers?
        # In the example 1,2,3,6:
        # Conflicts:
        # x=1: subsets {(2,3,6)}. y in {2,3,6}.
        # x=2: subsets {(3,6)}. y in {3,6}.
        # Conflict map: {2:1, 3:2, 6:2}.
        # Max count 2. Candidates: 3, 6.
        # If we remove 6: S={1,2,3,4..24}.
        #   Check x=1: 1/1 = 1/2+1/3+... ? 1/2+1/3 = 5/6 != 1. Valid.
        #   Check x=2: 1/2 = 1/3+... ? 1/3 != 1/2. Valid.
        #   So removing 6 works.
        # If we remove 3: S={1,2,4,6..24}.
        #   Check x=1: 1/1 = 1/2 + 1/6 + ... ? 1/2+1/6=2/3. Need 1/3. 1/4+1/12? If 4,12 in S.
        #   Check x=2: 1/2 = 1/4 + 1/4? No distinct. 1/2 = 1/4 + 1/6 + 1/12? 
        #   It might still be invalid.
        
        # So removing the element with highest conflict count is good.
        # Tie-breaker: Remove the largest y.
        y_to_remove = max(candidates)
        
        S.remove(y_to_remove)
        
    # Final check and potential local search (add back removed elements)
    removed = set(range(1, 25)) - S
    
    # Try to add back removed elements if they don't cause conflicts
    # Sort removed by value? 
    # Adding a small x is risky (large reciprocal, easy to sum).
    # Adding a large x is safer.
    for x in sorted(removed, reverse=True):
        if time.time() > deadline:
            break
        # Check if x can be added to S
        # This means 1/x must NOT be a subset sum of {1/y | y in S, y > x} 
        # AND 1/x must NOT be a subset sum of {1/y | y in S, y < x} (which is impossible as 1/y > 1/x)
        # So just check y > x.
        
        # Temporarily add x
        S.add(x)
        # Check validity of x
        valid = True
        # We only need to check if x is valid. Adding x might make smaller elements invalid?
        # No, x is larger than smaller elements. 
        # Smaller element z < x. 1/z > 1/x.
        # The sum for z uses y > z. x is one of them.
        # So adding x MIGHT make z invalid if 1/z = 1/x + sum(others).
        # So we must check all z in S, z < x.
        
        for z in S:
            if z >= x:
                continue
            # Check if 1/z is subset sum of {1/y | y in S, y > z, y != z}
            # Note: x is in S and x > z, so x is in the candidate pool.
            subsets = _find_conflicting_subsets(z, S)
            if subsets:
                valid = False
                break
        
        if valid:
            # Keep x
            pass
        else:
            # Remove x
            S.remove(x)
            
    return list(S)

# EVOLVE-BLOCK-END
