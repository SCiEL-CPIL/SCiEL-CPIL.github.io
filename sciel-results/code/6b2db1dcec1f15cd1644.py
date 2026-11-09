"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Precompute all subset-sum conflicts using bitsets
    # conflict[x] = bitmask of elements y such that 1/x can be formed as sum of 1/y over a subset of OTHER selected elements
    # Actually, we need: for each x, which subsets of [1..24]\{x} sum to 1/x
    # Then build a conflict graph: edge (x,y) exists if y is "necessary" for some subset summing to 1/x
    # But actually the condition is: 1/x != sum of reciprocals of ANY subset of S\{x}
    # This means: if there exists a subset T ⊆ S\{x} with sum(1/t for t in T) = 1/x, then S is invalid.
    # 
    # For MIS formulation: we need a graph where independent sets correspond to valid subsets.
    # Two elements x and y conflict if {x,y} is already invalid, OR if adding them together with others creates a conflict.
    # Actually, the constraint is global (subset sums), not pairwise. So MIS on a pairwise graph won't directly work.
    #
    # Alternative: Use branch-and-bound directly on the subset selection problem.
    # Key insight: process elements in a specific order, and use the fact that if 1/x is representable
    # by the reciprocals of already-selected larger elements, we can't include x.
    
    # Precompute: for each x in 1..24, find all subsets of [1..24]\{x} whose reciprocals sum to 1/x
    # This is expensive, so let's use a smarter approach.
    
    # Actually, let's think about it differently. The key observation is:
    # If we select elements in decreasing order, and at each step check if 1/elem can be formed
    # by the reciprocals of already-selected elements, that's not sufficient because we also need
    # to check that adding elem doesn't create a conflict for previously selected elements.
    
    # Let me use a different approach: precompute all "forbidden" subsets.
    # A subset T of [1..24] is "forbidden for x" if x ∉ T and sum(1/t for t in T) = 1/x.
    # Then S is valid iff for all x in S, no forbidden subset for x is contained in S\{x}.
    
    # Precompute forbidden subsets
    # For each x, find all T ⊆ [1..24]\{x} with sum(1/t) = 1/x
    # This is a subset sum problem with rational numbers.
    
    # Since the numbers are small (1..24), let's enumerate all subsets of [1..24] and check.
    # There are 2^24 = 16.7M subsets, which is too many.
    
    # Better: for each x, solve the subset sum problem on [1..24]\{x} targeting 1/x.
    # Use dynamic programming with Fractions, but prune aggressively.
    
    # Actually, let's just use the backtracking approach but with better pruning.
    # The current approach scores 1.5833 = 19/12, suggesting it finds sets of size ~19.
    # Let me try to improve the search.
    
    # New approach: Use a greedy + local search hybrid with better conflict detection.
    
    def _can_form_target(elements, target):
        """Check if any subset of elements' reciprocals sums to target.
        elements is a list of ints, target is a Fraction.
        Returns True if such a subset exists."""
        if not elements:
            return target == 0
        if target <= 0:
            return False
        
        # Use memoization with frozenset for small element sets
        # Sort descending for better pruning
        sorted_elems = sorted(elements, reverse=True)
        
        # DP: achievable sums using subset of first k elements
        achievable = {Fraction(0)}
        for elem in sorted_elems:
            recip = Fraction(1, elem)
            new_sums = set()
            for s in achievable:
                ns = s + recip
                if ns <= target:
                    new_sums.add(ns)
            achievable = achievable | new_sums
            if target in achievable:
                return True
        return False
    
    def is_valid_subset(S):
        """Check if subset S is valid (no reciprocal subsums)."""
        S_list = list(S)
        for a in S_list:
            target = Fraction(1, a)
            others = [b for b in S_list if b != a]
            if _can_form_target(others, target):
                return False
        return True
    
    # Strategy: Start with a large candidate set and remove elements that cause conflicts.
    # Then try to add back elements.
    
    # Phase 1: Greedy construction in descending order
    # For each element from 24 down to 1, add it if it doesn't create a conflict.
    
    def greedy_construct(order):
        """Build a valid set by processing elements in given order, adding if valid."""
        current = []
        for elem in order:
            candidate = current + [elem]
            if is_valid_subset(candidate):
                current = candidate
        return current
    
    # Try multiple orders
    best = []
    
    # Order 1: Descending
    candidate = greedy_construct(range(24, 0, -1))
    if len(candidate) > len(best):
        best = candidate
    
    # Order 2: Ascending
    candidate = greedy_construct(range(1, 25))
    if len(candidate) > len(best):
        best = candidate
    
    # Order 3: Primes first, then composites
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    composites = [x for x in range(1, 25) if x not in primes and x > 1]
    candidate = greedy_construct(primes + composites)
    if len(candidate) > len(best):
        best = candidate
    
    # Order 4: Larger numbers first (already tried)
    # Order 5: Try to include small numbers which are "harder" to form
    # Small numbers have large reciprocals, so they're harder to sum up.
    # Actually, 1/1 = 1, which is easy to form. 1/24 is hard.
    # So maybe ascending is better for small numbers.
    
    # Phase 2: Local search - try to add elements not in best
    best_set = set(best)
    remaining = [x for x in range(1, 25) if x not in best_set]
    
    improved = True
    while improved:
        improved = False
        for elem in remaining[:]:
            if time.time() > deadline - 5:
                break
            candidate = list(best) + [elem]
            if is_valid_subset(candidate):
                best = candidate
                best_set.add(elem)
                remaining.remove(elem)
                improved = True
    
    # Phase 3: Try swap-based improvement: remove one, add two
    best_set = set(best)
    for a in list(best_set):
        if time.time() > deadline - 5:
            break
        reduced = [x for x in best if x != a]
        for b in range(1, 25):
            if time.time() > deadline - 5:
                break
            if b in best_set:
                continue
            for c in range(b+1, 25):
                if time.time() > deadline - 5:
                    break
                if c in best_set:
                    continue
                candidate = reduced + [b, c]
                if len(candidate) > len(best):
                    if is_valid_subset(candidate):
                        best = candidate
                        best_set = set(best)
                        break
        if time.time() > deadline - 5:
            break
    
    # Phase 4: Backtracking search to find larger sets
    # Use the current best as a lower bound
    all_elements = list(range(24, 0, -1))
    
    def backtrack(index, current_set, best_size):
        nonlocal best
        if time.time() > deadline - 3:
            return
        
        # Prune: if we can't beat best
        if len(current_set) + (len(all_elements) - index) <= len(best):
            return
        
        if index == len(all_elements):
            if len(current_set) > len(best):
                best = list(current_set)
            return
        
        elem = all_elements[index]
        
        # Try including elem
        new_set = current_set + [elem]
        if is_valid_subset(new_set):
            backtrack(index + 1, new_set, best_size)
        
        # Try excluding elem
        backtrack(index + 1, current_set, best_size)
    
    # Only do backtracking if we have time
    if time.time() < deadline - 30:
        backtrack(0, [], len(best))
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return list(best)

# EVOLVE-BLOCK-END
