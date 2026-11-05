"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # LCM of 1..22 for exact integer arithmetic
    # LCM(1..22) = 232792560
    lcm_val = 1
    for i in range(1, 23):
        lcm_val = lcm_val * i // math.gcd(lcm_val, i)
    
    # Reciprocal of i scaled by LCM: lcm_val // i
    recip = [0] + [lcm_val // i for i in range(1, 23)]
    
    # We want to find the largest subset of [1..22] such that all subset sums are distinct
    # Use backtracking with pruning
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback: size 9
    
    # Try to find size 11 first, then 10, etc.
    # For a set S, all 2^|S| subset sums must be distinct
    # When adding x to S with subset sums SS, collision iff exists s1,s2 in SS with s1 - s2 = x (or s2 - s1 = x)
    
    def can_add(subset_sums, x):
        """Check if adding x to a set with given subset sums creates a collision."""
        # New sums are SS and SS + x. Collision if any two new sums are equal.
        # This happens iff there exist a, b in SS such that a = b + x, i.e., a - b = x
        # Or equivalently, x is in {a - b : a, b in SS, a != b}
        # Since we're using integers, check if x is a difference of two elements in SS
        ss_list = list(subset_sums)
        n = len(ss_list)
        # For each pair, check if difference is x
        # Optimization: for each a in SS, check if a - x is also in SS
        ss_set = subset_sums  # it's already a set
        for a in ss_list:
            if (a - x) in ss_set:
                return False
        return True
    
    def backtrack(remaining, current_sums, current_set, target_size):
        """Try to build a set of target_size from remaining elements."""
        nonlocal best_set
        if time.time() > deadline:
            return None
        
        if len(current_set) == target_size:
            return list(current_set)
        
        if not remaining:
            return None
        
        # We need to pick (target_size - len(current_set)) more from remaining
        needed = target_size - len(current_set)
        if len(remaining) < needed:
            return None
        
        # Try adding each element from remaining
        for i in range(len(remaining)):
            x = remaining[i]
            rx = recip[x]
            
            if can_add(current_sums, rx):
                # Compute new subset sums
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s)
                    new_sums.add(s + rx)
                
                # Recurse
                new_remaining = remaining[i+1:]
                result = backtrack(new_remaining, new_sums, current_set + [x], target_size)
                if result is not None:
                    return result
        
        return None
    
    # Try sizes from 11 down to 10
    all_elements = list(range(1, 23))
    
    for target_size in [11, 10]:
        if time.time() > deadline:
            break
        
        # Use a smarter ordering: try elements that are "large" first to reduce collisions
        # Actually, let's try different orderings
        # Strategy: try to find any valid set of the target size
        
        # To speed up, we can use a heuristic ordering
        # Try elements in decreasing order of their reciprocal value (i.e., increasing element)
        # Or try a fixed good ordering
        
        # Let's try multiple orderings
        orderings = [
            all_elements,  # natural order
            sorted(all_elements, reverse=True),  # reverse
        ]
        
        found = False
        for ordering in orderings:
            if time.time() > deadline:
                break
            if found:
                break
            
            result = backtrack(ordering, {0}, [], target_size)
            if result is not None:
                best_set = result
                found = True
                break
        
        if found:
            break
    
    # Verify the result
    if best_set:
        ss = {0}
        for x in best_set:
            rx = recip[x]
            new_ss = set()
            for s in ss:
                new_ss.add(s)
                new_ss.add(s + rx)
            if len(new_ss) != 2 * len(ss):
                # Collision detected, fallback to known good set
                best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
            ss = new_ss
    
    return best_set
# EVOLVE-BLOCK-END
