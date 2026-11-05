"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def _check_valid(selected):
    """Check if all unordered triples (with repetition) have unique sums."""
    n = len(selected)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = selected[i] + selected[j] + selected[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _greedy_construct(max_val=4096, target_size=80, time_limit=80.0):
    """
    Greedy construction with backtracking for B_3[1] sequences.
    We want to maximize the number of distinct positive integers <= max_val
    such that all unordered triples (with repetition) have unique sums.
    """
    start_time = time.time()
    
    # Start with a small seed set and grow it
    # Use a greedy approach: try to add the smallest possible number
    # that doesn't create a collision
    
    selected = []
    used_sums = set()  # all triple sums so far
    
    def can_add(x, selected, used_sums):
        """Check if x can be added without creating duplicate triple sums."""
        n = len(selected)
        new_sums = set()
        
        # Triples involving x:
        # (x, x, x)
        # (x, x, y) for y in selected
        # (x, y, y) for y in selected
        # (x, y, z) for y <= z in selected
        
        # (x, x, x)
        s = 3 * x
        if s in used_sums or s in new_sums:
            return False
        new_sums.add(s)
        
        # (x, x, y)
        for y in selected:
            s = 2 * x + y
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x, y, y)
        for y in selected:
            s = x + 2 * y
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
        
        # (x, y, z) where y <= z
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
                if s in used_sums or s in new_sums:
                    return False
                new_sums.add(s)
        
        return True
    
    # Greedy: try adding smallest available number
    for x in range(1, max_val + 1):
        if time.time() - start_time > time_limit:
            break
        if len(selected) >= target_size:
            break
        if can_add(x, selected, used_sums):
            # Add x and update used_sums
            n = len(selected)
            s = 3 * x
            used_sums.add(s)
            for y in selected:
                used_sums.add(2 * x + y)
                used_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i, n):
                    used_sums.add(x + selected[i] + selected[j])
            selected.append(x)
    
    return selected

def _local_search_improve(selected, max_val=4096, time_limit=80.0):
    """
    Try to improve the set by replacing elements or trying different orders.
    This is a heuristic local search.
    """
    start_time = time.time()
    current = list(selected)
    current_size = len(current)
    
    # Try to add more elements by removing some and re-adding
    # Or try different greedy orders
    
    # First, try to see if we can add more elements
    used_sums = set()
    n = len(current)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                used_sums.add(current[i] + current[j] + current[k])
    
    # Try to add more elements
    for x in range(1, max_val + 1):
        if time.time() - start_time > time_limit:
            break
        if x in current:
            continue
        if _can_add_fast(x, current, used_sums):
            n = len(current)
            used_sums.add(3 * x)
            for y in current:
                used_sums.add(2 * x + y)
                used_sums.add(x + 2 * y)
            for i in range(n):
                for j in range(i, n):
                    used_sums.add(x + current[i] + current[j])
            current.append(x)
            current_size += 1
    
    return current

def _can_add_fast(x, selected, used_sums):
    """Fast check if x can be added."""
    n = len(selected)
    
    # (x, x, x)
    if 3 * x in used_sums:
        return False
    
    # (x, x, y)
    for y in selected:
        if 2 * x + y in used_sums:
            return False
    
    # (x, y, y)
    for y in selected:
        if x + 2 * y in used_sums:
            return False
    
    # (x, y, z)
    for i in range(n):
        for j in range(i, n):
            if x + selected[i] + selected[j] in used_sums:
                return False
    
    return True

def construct():
    """
    Construct a set of up to 80 distinct positive integers <= 4096
    where all unordered triples (with repetition) have unique sums.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Start with greedy construction
    result = _greedy_construct(max_val=4096, target_size=80, time_limit=40.0)
    
    if time.time() > deadline:
        return result
    
    # Try to improve with local search
    improved = _local_search_improve(result, max_val=4096, time_limit=15.0)
    
    if time.time() > deadline:
        return result
    
    # Return the better result
    if len(improved) >= len(result):
        return improved
    return result

# EVOLVE-BLOCK-END
