"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _verify_reciprocal_distinct(nums):
    """Check that all subset sums of 1/n are distinct."""
    n = len(nums)
    # Use fractions to be exact
    from fractions import Fraction
    seen = set()
    # 2^n subset sums
    # For n up to ~15, 2^15 = 32768, manageable
    # We'll use bitmask enumeration
    for mask in range(1 << n):
        s = Fraction(0, 1)
        for i in range(n):
            if mask & (1 << i):
                s += Fraction(1, nums[i])
        if s in seen:
            return False
        seen.add(s)
    return True

def _is_dissociated(nums):
    """A set is dissociated if all subset sums are distinct."""
    return _verify_reciprocal_distinct(nums)

def construct():
    deadline = time.time() + 50.0  # safety margin before 60s limit
    start_time = time.time()
    
    # The seed returns 9 elements. We want to maximize cardinality.
    # Known: {1,2,3,5,7,11,13,17,19} works (9 elements).
    # Can we get 10 or more?
    # 
    # Strategy: Greedy construction with backtracking.
    # Start with a candidate set and try to add elements.
    # 
    # Key insight: For reciprocal sums to be distinct, the set must be "dissociated".
    # A sufficient condition: each element is larger than the sum of all smaller elements' reciprocals
    # times some factor... actually the condition is that no element's reciprocal equals a sum of others.
    # 
    # Better approach: Use the fact that if we pick numbers such that each is "large enough",
    # the subset sums are distinct. Specifically, if 1/a_i > sum_{j<i} 1/a_j, then all subset sums are distinct.
    # This is the "superincreasing" condition for reciprocals, i.e., the reciprocals form a superincreasing sequence.
    # 
    # Let's try: pick a_1, a_2, ..., a_k such that 1/a_i > sum_{j=1}^{i-1} 1/a_j.
    # This means a_i < 1 / (sum_{j=1}^{i-1} 1/a_j).
    # Wait, that's not right. If 1/a_i > sum_{j<i} 1/a_j, then a_i < 1/(sum_{j<i} 1/a_j).
    # But we want a_i to be in [1,22], so we need a_i >= 1.
    # 
    # Actually, the condition for distinct subset sums is: for each i, 1/a_i is not equal to any sum of 1/a_j for j != i.
    # A sufficient condition is that the reciprocals are linearly independent over {-1,0,1}, which is strong.
    # 
    # Let me try a different approach: brute force search for the largest valid subset.
    # Since 22 choose k for large k is expensive, let's use a greedy + local search.
    
    # First, let's try to extend the known 9-element set.
    base = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try adding each remaining element from [4,6,8,9,10,12,14,15,16,18,20,21,22]
    remaining = [x for x in range(1, 23) if x not in base]
    
    best = list(base)
    
    # Try to add one element
    for x in remaining:
        candidate = sorted(base + [x])
        if _is_dissociated(candidate):
            best = candidate
            break
    
    # Try to add two elements
    if len(best) == 9:
        remaining2 = [x for x in range(1, 23) if x not in best]
        for i in range(len(remaining2)):
            for j in range(i+1, len(remaining2)):
                candidate = sorted(best + [remaining2[i], remaining2[j]])
                if _is_dissociated(candidate):
                    best = candidate
                    break
            if len(best) > 9:
                break
            if time.time() > deadline - 10:
                break
    
    # Try to add three elements
    if len(best) < 11:
        remaining3 = [x for x in range(1, 23) if x not in best]
        if len(remaining3) >= 3:
            for i in range(len(remaining3)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(remaining3)):
                    if time.time() > deadline - 5:
                        break
                    for k in range(j+1, len(remaining3)):
                        candidate = sorted(best + [remaining3[i], remaining3[j], remaining3[k]])
                        if _is_dissociated(candidate):
                            best = candidate
                            break
                    if len(best) > len(best) - 1:  # if we found one
                        break
                if len(best) >= 12:
                    break
    
    # Now try a more systematic search: start from scratch with greedy
    # Greedy: add elements in order of increasing value, checking validity
    def greedy_construct(start_set, elements_to_try):
        current = list(start_set)
        for x in elements_to_try:
            if x in current:
                continue
            candidate = current + [x]
            if _is_dissociated(candidate):
                current = candidate
        return current
    
    # Try different orderings
    import random
    random.seed(42)
    
    # Try greedy with different orderings
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 23))
        random.shuffle(order)
        result = greedy_construct([], order)
        if len(result) > len(best):
            best = result
    
    # Local search: try to swap elements
    for _ in range(100):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(best)):
            if time.time() > deadline - 3:
                break
            for x in range(1, 23):
                if x in best:
                    continue
                candidate = [v for v in best if v != best[i]] + [x]
                if _is_dissociated(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    # Final verification
    if not _is_dissociated(best):
        # Fallback to known good set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return sorted(best)
# EVOLVE-BLOCK-END
