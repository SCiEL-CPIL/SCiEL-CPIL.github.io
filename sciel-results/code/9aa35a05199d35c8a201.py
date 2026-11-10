import time
from math import gcd
from functools import reduce
from itertools import combinations
from fractions import Fraction

def _lcm(a, b):
    return a * b // gcd(a, b)

def _check_element_valid(x, S):
    """
    Check if x is valid in set S (which includes x).
    Returns True if 1/x is NOT a subset sum of {1/y : y in S, y != x}.
    """
    target = Fraction(1, x)
    others = [y for y in S if y != x]
    m = len(others)
    
    if m == 0:
        return True
    
    if m <= 18:
        # Direct enumeration
        for r in range(1, m + 1):
            for subset in combinations(others, r):
                s = sum(Fraction(1, y) for y in subset)
                if s == target:
                    return False
        return True
    else:
        # Meet in the middle
        half = m // 2
        left = others[:half]
        right = others[half:]
        
        left_sums = set()
        for mask in range(1, 1 << len(left)):
            s = Fraction(0)
            for j in range(len(left)):
                if mask & (1 << j):
                    s += Fraction(1, left[j])
            left_sums.add(s)
        
        for mask in range(1 << len(right)):
            s_right = Fraction(0)
            for j in range(len(right)):
                if mask & (1 << j):
                    s_right += Fraction(1, right[j])
            
            needed = target - s_right
            if needed in left_sums:
                if mask == 0 and needed == 0:
                    continue
                return False
        
        return True

def _can_add(x, S):
    """
    Check if x can be added to S without violating the property.
    S does not contain x.
    Returns True if S ∪ {x} is valid.
    """
    S_with_x = S | {x}
    
    # Check x itself
    if not _check_element_valid(x, S_with_x):
        return False
    
    # Check all y in S that are less than x
    for y in S:
        if y < x:
            if not _check_element_valid(y, S_with_x):
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Branch and bound search
    # Order elements to try larger ones first (they're "safer" to include)
    candidates = list(range(24, 0, -1))
    
    best_set = set()
    best_size = 0
    
    # Greedy initial solution for strong lower bound
    greedy_set = set()
    for x in candidates:
        if time.time() > deadline:
            break
        if _can_add(x, greedy_set):
            greedy_set.add(x)
    
    best_set = set(greedy_set)
    best_size = len(greedy_set)
    
    # Now do branch and bound
    def branch_and_bound(idx, current_set, remaining_count):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current size + remaining elements
        upper_bound = len(current_set) + remaining_count
        if upper_bound <= best_size:
            return
        
        if idx >= len(candidates):
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
            return
        
        x = candidates[idx]
        new_remaining = remaining_count - 1
        
        # Option 1: Include x
        if _can_add(x, current_set):
            current_set.add(x)
            branch_and_bound(idx + 1, current_set, new_remaining)
            current_set.remove(x)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, current_set, new_remaining)
    
    branch_and_bound(0, set(), len(candidates))
    
    return list(best_set)

# EVOLVE-BLOCK-END
