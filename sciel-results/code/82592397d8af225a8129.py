"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def _subset_sum_exists(elements, target):
        """Check if any subset of elements' reciprocals sums to target."""
        n = len(elements)
        if n == 0:
            return target == 0
        
        # Use a set of achievable sums with pruning
        achievable = {Fraction(0)}
        for elem in elements:
            recip = Fraction(1, elem)
            new_achievable = set()
            for s in achievable:
                ns = s + recip
                if ns <= target:
                    new_achievable.add(ns)
            achievable = achievable | new_achievable
            if target in achievable:
                return True
        return target in achievable
    
    def is_valid_subset(S):
        """Check if subset S is valid (no reciprocal subsums)."""
        S_list = list(S)
        S_set = set(S)
        for a in S_list:
            target = Fraction(1, a)
            others = [b for b in S_list if b != a]
            if _subset_sum_exists(others, target):
                return False
        return True
    
    # Systematic backtracking search with forward checking
    # Process elements in descending order (larger first = safer)
    all_elements = list(range(24, 0, -1))
    
    best = []
    
    def backtrack(index, current_set):
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
        # Quick check: can 1/elem be formed by current_set?
        if not _subset_sum_exists(current_set, Fraction(1, elem)):
            # Check if adding elem creates conflicts for existing elements
            valid = True
            for a in current_set:
                target_without = Fraction(1, a) - Fraction(1, elem)
                if target_without > 0:
                    others = [b for b in current_set if b != a]
                    if _subset_sum_exists(others, target_without):
                        valid = False
                        break
            if valid:
                backtrack(index + 1, new_set)
        
        # Try excluding elem
        backtrack(index + 1, current_set)
    
    backtrack(0, [])
    
    # Local search: try to add remaining elements
    best_set = set(best)
    remaining = [x for x in range(1, 25) if x not in best_set]
    
    for elem in remaining:
        if time.time() > deadline - 5:
            break
        candidate = list(best) + [elem]
        if is_valid_subset(candidate):
            best = candidate
            best_set.add(elem)
            remaining.remove(elem)
    
    # Try swap-based improvement
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
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return list(best)

# EVOLVE-BLOCK-END
