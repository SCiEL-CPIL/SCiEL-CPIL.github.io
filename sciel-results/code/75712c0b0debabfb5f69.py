"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}

def _is_valid(S_set):
    """Check if set S_set has no forbidden triple."""
    S = list(S_set)
    S.sort()
    n = len(S)
    for i in range(n):
        for j in range(i+1, n):
            a, b = S[i], S[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def _count_violations(S_set):
    """Count number of forbidden triples (for optimization)."""
    S = list(S_set)
    S.sort()
    n = len(S)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = S[i], S[j]
            ab = a + b
            if ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S[k]
                if a + c in S_set and b + c in S_set:
                    count += 1
    return count

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = P['n']
    # Start with the known valid solution: {65, 66, ..., 128}
    best = set(range(n//2 + 1, n + 1))
    best_size = len(best)
    
    # Precompute: for each pair (a,b) in 1..128, what's a+b
    # We'll work with the full universe 1..128
    
    # Strategy: try to add elements from 1..64 to the base set,
    # but we may need to remove some elements from 65..128 to maintain validity.
    
    # Alternative: local search on the full set 1..128
    # Start from {65..128} and try swaps or additions
    
    # Let's try a greedy approach: for each element in 1..64, 
    # check if adding it (and possibly removing conflicting elements) helps.
    
    # First, let's try simple local search:
    # State: a set S. Moves: add an element, remove an element, swap.
    # Objective: maximize |S| subject to validity.
    
    current = set(best)
    current_size = best_size
    
    # Precompute for efficiency
    all_elements = list(range(1, n + 1))
    
    # Try to improve by adding elements from 1..64
    # For each candidate element x in 1..64:
    #   Check if current ∪ {x} is valid. If yes, add it.
    #   If not, check if we can remove some elements to make it valid and still gain net size.
    
    # Simpler: try all single additions first
    for x in range(1, n//2 + 1):
        if x in current:
            continue
        new_set = current | {x}
        if _is_valid(new_set):
            current = new_set
            current_size += 1
    
    if current_size > best_size:
        best = set(current)
        best_size = current_size
    
    # Now try swaps: replace one element in current with one not in current
    # This is expensive, so limit it
    if time.time() < deadline - 10:
        # Try to find a better configuration via local search
        # We'll try removing one element and adding two (net +1)
        # Or just try single swaps for improvement
        
        # Actually, let's try a different approach:
        # The structure suggests that if we take elements in (64, 128], we're safe.
        # If we take elements in (32, 64], their sums are in (64, 128], 
        # so we need to be careful.
        
        # Let me try: S = {x : x > 64} ∪ {x : 32 < x <= 64 and x+x > 128}
        # x+x > 128 means x > 64, so that's empty.
        
        # What about S = {x : x > 64} ∪ {x : 43 < x <= 64}?
        # For a,b in (43,64]: a+b in (86, 128]. These could be in S if > 64.
        # For a,b,c all in (43,64]: a+b, a+c, b+c all in (86,128].
        # If any of these sums are in S (i.e., in (64,128]), we have a problem.
        
        # This is getting complex. Let me just do a proper local search.
        
        # Local search: try all single-element additions and removals
        # Iterate until no improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try additions
            for x in range(1, n + 1):
                if x in current:
                    continue
                new_set = current | {x}
                if _is_valid(new_set):
                    current = new_set
                    current_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try removals that allow additions (swap)
            # For each element to remove, try adding each element not in set
            for rem in list(current):
                if time.time() > deadline - 5:
                    break
                temp_set = current - {rem}
                for add in range(1, n + 1):
                    if add in temp_set:
                        continue
                    new_set = temp_set | {add}
                    if len(new_set) > len(current) and _is_valid(new_set):
                        current = new_set
                        current_size = len(current)
                        improved = True
                        break
                if improved:
                    break
        
        if current_size > best_size:
            best = set(current)
            best_size = current_size
    
    # Final: try to add more elements greedily
    for x in range(1, n + 1):
        if time.time() > deadline - 2:
            break
        if x in best:
            continue
        new_set = best | {x}
        if _is_valid(new_set):
            best = new_set
            best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
