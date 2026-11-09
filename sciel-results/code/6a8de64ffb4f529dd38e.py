"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
from functools import reduce
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_dissociated_fast(subset):
    """
    Check if all reciprocal subset sums are distinct.
    Uses LCM to convert to integer subset sum problem.
    """
    if not subset:
        return True
    
    # Calculate LCM of all elements in subset
    L = 1
    for x in subset:
        L = _lcm(L, x)
        if L > 10**18: # Avoid huge numbers if possible, though for N=22 it's fine
             # Actually LCM of 1..22 is ~ 232792560, which fits in int64.
             pass
            
    # Convert to integer values
    vals = [L // x for x in subset]
    
    # Generate all subset sums of vals
    # Use a set to check for uniqueness
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            new_sums.add(s + v)
        # If we added v, the size should double. 
        # If any collision happens, size won't double.
        if len(new_sums) != 2 * len(sums):
            return False
        sums = new_sums
        
    return True

def construct():
    # Strategy: 
    # 1. We know that "large" numbers are more likely to be dissociated because their reciprocals are small and "spread out" less prone to small integer relations? 
    #    Actually, small integers like 1, 2, 4 are powers of 2 and are dissociated.
    #    Large integers like 20, 21, 22 have reciprocals 1/20, 1/21, 1/22.
    #    Let's test if the set of all large numbers is dissociated.
    
    # We will search for the maximum cardinality.
    # We start with the set of all numbers 1..22 and try to remove the minimum number of elements to make it dissociated?
    # Or we build up from a good candidate.
    
    # Candidate 1: Powers of 2: {1, 2, 4, 8, 16} -> Size 5.
    # Candidate 2: Large numbers: {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22} -> Size 11.
    
    # Let's verify if {12..22} is dissociated.
    # If it is, we can try to add 11, 10, etc.
    
    best_set = set()
    best_size = 0
    
    # Try all subsets of [1, 22] with size k, from high to low.
    # But 2^22 is too many.
    # However, we can use a heuristic: 
    # Iterate through sizes from 12 down to 1.
    # For each size, try a few specific sets or use a greedy construction.
    
    # Greedy Construction with Backtracking/Local Search using the fast checker.
    
    # Start with the set of largest numbers, as they are "independent" in terms of magnitude.
    # Let's try to build a set by adding elements from 22 down to 1.
    
    # We will do a DFS-like search or just a simple greedy with swaps.
    
    # Let's try a randomized local search with the fast checker.
    # It is very fast now.
    
    import random
    random.seed(42)
    
    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    for _ in range(50):
        # Random initialization: pick a random subset of size ~10
        # Or start with {12..22}
        if _ == 0:
            current = set(range(12, 23))
        else:
            current = set(random.sample(range(1, 23), 10))
            
        if not _is_dissociated_fast(current):
             # If init is bad, shrink it until it is good? 
             # Or just start empty and add?
             # Let's start empty and add greedily.
             current = set()
             for x in range(22, 0, -1):
                 if _is_dissociated_fast(current | {x}):
                     current.add(x)
        
        # Local Search: Try to add elements, then swap.
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any element from 1..22 not in current
            added = False
            for x in range(1, 23):
                if x in current:
                    continue
                if _is_dissociated_fast(current | {x}):
                    current.add(x)
                    added = True
                    improved = True
                    break # Restart loop as set changed
            
            # 2. If no add, try swaps
            if not added:
                # Try swapping x in current with y not in current
                # Only if it leads to a state where we can add more? 
                # Or just accept if it's dissociated and we hope to add later?
                # Swaps maintain size. We want to increase size.
                # So we only swap if it allows a subsequent add.
                
                # To be efficient, we just try a few random swaps
                for _ in range(20):
                    if not current: break
                    x = random.choice(list(current))
                    y = random.choice([c for c in range(1, 23) if c not in current])
                    new_set = (current - {x}) | {y}
                    if _is_dissociated_fast(new_set):
                        # Check if this new set allows adding an element
                        can_add = False
                        for z in range(1, 23):
                            if z in new_set: continue
                            if _is_dissociated_fast(new_set | {z}):
                                can_add = True
                                break
                        if can_add:
                            current = new_set
                            improved = True
                            break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current.copy()
            
    # Fallback: If we didn't find anything (shouldn't happen), return a known good set
    if not best_set:
        return [1, 2, 4, 8, 16]
        
    return list(best_set)

# EVOLVE-BLOCK-END
