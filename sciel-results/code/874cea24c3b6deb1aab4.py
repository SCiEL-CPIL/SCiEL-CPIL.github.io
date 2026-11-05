"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # The key insight from the knowledge:
    # When adding elements in decreasing order (from large to small),
    # a new element i (smaller) has reciprocal 1/i which is LARGER than
    # reciprocals of already-selected elements j > i.
    # 
    # For a new element i to be "subsumed", we'd need:
    #   1/i = sum of 1/j for some subset of already-selected {j > i}
    # But since all j > i, we have 1/j < 1/i, so the sum of any subset
    # of {1/j : j > i} could potentially equal 1/i.
    # 
    # However, the critical observation is about the OTHER direction:
    # When we add a new small element i, could it cause an existing element j > i
    # to become subsumed? That would require:
    #   1/j = sum of 1/k for some subset of OTHER selected elements including i
    # Since i < j, 1/i > 1/j, so any subset containing i has sum > 1/i > 1/j.
    # Thus, adding smaller elements NEVER invalidates larger existing elements!
    #
    # So we can greedily add elements from largest to smallest:
    # - Start with empty set
    # - For i from 24 down to 1:
    #   Check if 1/i can be formed by a subset of reciprocals of currently selected elements
    #   If NOT, add i to the set
    #
    # This gives a maximal (not necessarily maximum) set, but it's fast.
    # Then we can try to improve with local search.
    
    def can_form_target(target_frac, selected_set):
        """Check if target_frac can be formed by a non-empty subset of reciprocals of selected_set."""
        if not selected_set:
            return False
        
        elems = sorted(selected_set, reverse=True)  # Largest elements first (smallest reciprocals)
        n = len(elems)
        
        # If sum of all reciprocals < target, impossible
        total = sum(Fraction(1, y) for y in elems)
        if total < target_frac:
            return False
        
        # If any single element's reciprocal equals target, and that element is in the set
        # (but we need a SUBSET, so a single element subset is valid)
        for y in elems:
            if Fraction(1, y) == target_frac:
                return True
        
        # DFS with pruning
        # Sort by reciprocal descending (i.e., element ascending) for better pruning
        elems_asc = sorted(selected_set)  # Smallest elements first (largest reciprocals)
        recips = [Fraction(1, y) for y in elems_asc]
        suffix_sums = [0] * (len(recips) + 1)
        for idx in range(len(recips) - 1, -1, -1):
            suffix_sums[idx] = suffix_sums[idx + 1] + recips[idx]
        
        found = [False]
        
        def dfs(idx, acc):
            if found[0]:
                return
            if acc == target_frac:
                found[0] = True
                return
            if idx >= len(recips):
                return
            # Prune: even taking all remaining can't reach target
            if acc + suffix_sums[idx] < target_frac:
                return
            # Prune: already exceeded
            if acc > target_frac:
                return
            
            # Try including recips[idx]
            dfs(idx + 1, acc + recips[idx])
            # Try excluding recips[idx]
            dfs(idx + 1, acc)
        
        dfs(0, Fraction(0))
        return found[0]
    
    # Greedy: add from largest to smallest
    selected = set()
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        target = Fraction(1, i)
        if not can_form_target(target, selected):
            selected.add(i)
    
    best = set(selected)
    best_size = len(best)
    
    # Try to improve with local search: try removing one element and adding others
    # Or try different greedy orders
    
    # Alternative greedy: try all possible first elements
    for first in range(24, 0, -1):
        if time.time() > deadline:
            break
        
        selected2 = {first}
        for i in range(24, 0, -1):
            if i == first:
                continue
            if time.time() > deadline:
                break
            target = Fraction(1, i)
            if not can_form_target(target, selected2):
                selected2.add(i)
        
        if len(selected2) > best_size:
            best = set(selected2)
            best_size = len(selected2)
    
    # Local search: try to swap elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best)
        for remove in list(current):
            if time.time() > deadline:
                break
            for add in range(24, 0, -1):
                if add in current:
                    continue
                candidate = set(current)
                candidate.remove(remove)
                candidate.add(add)
                # Validate the entire set
                valid = True
                for elem in candidate:
                    target = Fraction(1, elem)
                    others = candidate - {elem}
                    if can_form_target(target, others):
                        valid = False
                        break
                if valid and len(candidate) > len(current):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
    
    # Try adding elements one by one to the current best
    for i in range(24, 0, -1):
        if time.time() > deadline:
            break
        if i in best:
            continue
        candidate = set(best)
        candidate.add(i)
        # Validate
        valid = True
        for elem in candidate:
            target = Fraction(1, elem)
            others = candidate - {elem}
            if can_form_target(target, others):
                valid = False
                break
        if valid:
            best = candidate
    
    # Convert to sorted list
    result = sorted(best)
    
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result

# EVOLVE-BLOCK-END
