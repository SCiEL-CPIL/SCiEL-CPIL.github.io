"""Rewritten program: efficient search for 3x3 sum grids with pairwise-coprime sums.

Key insight: all sums must be odd (A all odd, B all even or vice versa).
We need 9 distinct pairwise-coprime odd numbers formable as a_i + b_j.
"""
import time
import itertools
import random
from math import gcd

def _pairwise_coprime(nums):
    """Check if all values in nums are pairwise coprime."""
    n = len(nums)
    for i in range(n):
        for j in range(i+1, n):
            if gcd(nums[i], nums[j]) != 1:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    
    # All odd numbers in [1,128]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    # All even numbers in [2,128]
    evens = [x for x in range(2, 129) if x % 2 == 0]
    
    best_count = 0
    best_result = ([1, 3, 5], [2, 4, 6])
    
    random.seed(42)
    
    A_combos = list(itertools.combinations(odds, 3))
    
    for A in A_combos:
        if time.time() > deadline:
            break
        
        a0, a1, a2 = A
        # A differences
        A_diffs = {abs(a1-a0), abs(a2-a0), abs(a2-a1)}
        
        # Try random B's with difference filter
        for _ in range(300):
            if time.time() > deadline:
                break
            B = tuple(sorted(random.sample(evens, 3)))
            b0, b1, b2 = B
            B_diffs = {b1-b0, b2-b0, b2-b1}
            if B_diffs & A_diffs:  # Common difference -> duplicate sums
                continue
            
            # Check pairwise coprime
            sums = [a + b for a in A for b in B]
            if len(set(sums)) == 9 and _pairwise_coprime(sums):
                best_count = 9
                best_result = (list(A), list(B))
                return list(best_result[0]), list(best_result[1])
    
    return list(best_result[0]), list(best_result[1])
