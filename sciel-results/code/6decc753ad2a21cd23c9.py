"""
Strategy: Greedy Construction with Exact Rational Verification.

The current seed program returns a fixed set [13..24]. This strategy actively searches
for a larger valid set by:
1. Defining a validity check using exact rational arithmetic (Fraction) to avoid FP errors.
2. Using a meet-in-the-middle algorithm for the subset-sum check to ensure efficiency.
3. Running multiple greedy construction passes with different ordering heuristics
   (descending, ascending, random) to find the maximum cardinality set.
4. Enforcing a strict time deadline to return the best found set safely.
"""

import time
import random
from fractions import Fraction
from itertools import combinations

# Global state for deadline
_START_TIME = None
DEADLINE = 75.0  # Safety margin for 90s limit

def _time_remaining():
    global _START_TIME
    if _START_TIME is None:
        return DEADLINE
    return DEADLINE - (time.time() - _START_TIME)

def _subset_sum_possible(target, values):
    """
    Check if target can be formed by a subset of values.
    Uses meet-in-the-middle for efficiency when len(values) is moderate.
    values: list of Fractions
    target: Fraction
    """
    n = len(values)
    if n == 0:
        return target == 0
    
    # If n is small, just brute force
    if n <= 20:
        for i in range(1, 1 << n):
            s = Fraction(0)
            for j in range(n):
                if i & (1 << j):
                    s += values[j]
            if s == target:
                return True
        return False
    
    # Meet in the middle
    mid = n // 2
    left = values[:mid]
    right = values[mid:]
    
    # Generate all subset sums for left
    left_sums = set()
    for i in range(1 << len(left)):
        s = Fraction(0)
        for j in range(len(left)):
            if i & (1 << j):
                s += left[j]
        left_sums.add(s)
        
    # Generate all subset sums for right and check
    for i in range(1 << len(right)):
        s = Fraction(0)
        for j in range(len(right)):
            if i & (1 << j):
                s += right[j]
        # Check if target - s is in left_sums
        if (target - s) in left_sums:
            return True
            
    return False

def is_valid_candidate(selected):
    """
    Check if the set 'selected' is reciprocal-subsum-free.
    Condition: For each x in selected, 1/x != sum(1/y for y in subset of selected\{x})
    """
    if not selected:
        return False
        
    reciprocals = {x: Fraction(1, x) for x in selected}
    
    for x in selected:
        target = reciprocals[x]
        others = [reciprocals[y] for y in selected if y != x]
        
        if _subset_sum_possible(target, others):
            return False
            
    return True

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = list(range(13, 25)) # Fallback to seed
    best_size = len(best_set)
    
    # Ensure seed is valid (it should be, but let's check once)
    if not is_valid_candidate(best_set):
        # If seed is invalid, we must find a valid one.
        # A single element is always valid.
        best_set = [1]
        best_size = 1

    # Strategies to try
    # 1. Descending order greedy (starts with large numbers, smaller reciprocals)
    # 2. Ascending order greedy
    # 3. Random permutations
    
    orders = []
    orders.append(list(range(24, 0, -1)))
    orders.append(list(range(1, 25)))
    
    # Add some random permutations if time permits
    random.seed(42)
    for _ in range(10):
        perm = list(range(1, 25))
        random.shuffle(perm)
        orders.append(perm)
        
    # Also try starting with specific "hard" elements
    # Large numbers have small reciprocals, making it harder to sum them up.
    # Small numbers have large reciprocals.
    
    for order in orders:
        if _time_remaining() < 2.0:
            break
            
        current_set = []
        
        for x in order:
            if _time_remaining() < 0.5:
                break
                
            # Tentatively add x
            new_set = current_set + [x]
            
            # Check validity of new_set
            # Optimization: Only check constraints involving x?
            # If current_set is valid, adding x only creates new violations if:
            # 1. 1/x == sum(1/y for y in subset of current_set)
            # 2. For some y in current_set, 1/y == sum(1/z for z in subset of new_set\{y})
            #    The second case implies 1/y was a sum of a subset of current_set\{y} PLUS possibly x?
            #    No, the subset of OTHERS in new_set\{y} can include x.
            #    So 1/y == sum(1/z for z in S) where S is subset of current_set\{y} U {x}.
            #    This means either:
            #    a) x not in S: then 1/y was already a sum in current_set (which is valid, so false)
            #    b) x in S: then 1/y - 1/x = sum(1/z for z in S\{x}) where S\{x} is subset of current_set\{y}
            #    
            # So we only need to check:
            # 1. Does 1/x equal sum of a subset of reciprocals of current_set?
            # 2. For each y in current_set, does 1/y - 1/x equal sum of a subset of reciprocals of current_set\{y}?
            
            if not current_set:
                current_set = new_set
                continue
                
            # Check 1: Is 1/x a subset sum of current_set?
            others_for_x = [Fraction(1, y) for y in current_set]
            if _subset_sum_possible(Fraction(1, x), others_for_x):
                continue # Cannot add x
            
            # Check 2: For each y in current_set, does adding x break y's constraint?
            # We need to check if 1/y - 1/x is a subset sum of {1/z for z in current_set if z != y}
            valid_add = True
            for y in current_set:
                target_y_minus_x = Fraction(1, y) - Fraction(1, x)
                if target_y_minus_x < 0:
                    # Sum of positive fractions cannot be negative, so this specific combination is impossible.
                    # But wait, the subset sum could be empty? No, target must be matched exactly.
                    # If target < 0, it's impossible.
                    continue
                others_for_y = [Fraction(1, z) for z in current_set if z != y]
                if _subset_sum_possible(target_y_minus_x, others_for_y):
                    valid_add = False
                    break
            
            if valid_add:
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we found a very large set, we might stop early?
            # Max possible is 24.
            if best_size == 24:
                break

    # Final safety check
    if not is_valid_candidate(best_set):
        # Fallback to a known valid set
        # [1] is always valid
        return [1]
        
    return best_set

# Keep the function name exactly as requested
# The prompt asks to keep function names and signatures.
# The seed has: def construct():
# So we just expose construct.

if __name__ == "__main__":
    res = construct()
    print(res)
