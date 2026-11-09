"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from itertools import combinations

def _is_dissociated(subset):
    """
    Check if all reciprocal subset sums are distinct.
    Uses itertools to generate all subset sums and checks for uniqueness.
    This is O(2^k) which is fast for k <= 12.
    """
    if not subset:
        return True
    
    sums = set()
    # Start with 0 (empty subset)
    sums.add(Fraction(0))
    
    # Generate all subset sums
    # For k elements, there are 2^k subsets.
    # We can generate them iteratively or use combinations.
    # Iterative is faster for sets.
    current_sums = {Fraction(0)}
    for x in subset:
        inv_x = Fraction(1, x)
        new_sums = set()
        for s in current_sums:
            new_sums.add(s)
            new_sums.add(s + inv_x)
        # Early exit if collision detected
        if len(new_sums) != 2 * len(current_sums):
            return False
        current_sums = new_sums
        
    return True

def construct():
    # Strategy: Deterministic Greedy with Local Search.
    # 1. Start with a known good structure (powers of 2).
    # 2. Try to add more elements from [1, 22].
    # 3. Use local search (swaps) to escape local minima if we get stuck.
    
    # Initial set: Powers of 2 up to 16.
    # {1, 2, 4, 8, 16} is dissociated.
    current_set = {1, 2, 4, 8, 16}
    
    # List of all candidates
    all_candidates = list(range(1, 23))
    
    # Greedy Phase: Try to add elements one by one.
    # Order: Try numbers that are "safe". 
    # Heuristic: Numbers with large prime factors are safer.
    # Let's just try in descending order first, then ascending, then random?
    # To be deterministic and robust, let's try adding any available element.
    
    # We will run a loop that tries to improve the set size.
    # If we can't add any, we try swaps.
    
    max_size = 0
    best_set = set()
    
    # We'll do a few restarts with different initializations to be safe.
    # Init 1: Powers of 2
    # Init 2: Random
    # Init 3: Large numbers
    
    initializations = [
        [1, 2, 4, 8, 16],
        [2, 3, 5, 7, 11, 13, 17, 19], # Primes
        list(range(22, 11, -1)), # Large numbers
        list(range(1, 13)) # Small numbers
    ]
    
    for init_list in initializations:
        # Filter to valid subset of [1, 22]
        S = set(x for x in init_list if 1 <= x <= 22)
        
        # If the init itself is not dissociated, prune it
        if not _is_dissociated(S):
             # Try to repair by removing elements? 
             # Simpler: just skip this init if it's bad, or reduce it.
             # Let's just reduce it to the largest dissociated subset.
             # This is expensive. Let's just try to add to it anyway? No, _is_dissociated will fail.
             # Let's just start with empty if init is bad?
             # Actually, let's just try to build from scratch if init is bad.
             # For now, skip bad inits.
             continue
        
        current_size = len(S)
        
        # Local Search Loop
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add an element
            added = False
            for x in all_candidates:
                if x in S:
                    continue
                new_S = S | {x}
                if _is_dissociated(new_S):
                    S = new_S
                    added = True
                    improved = True
                    break # Restart the addition loop as the set changed
            
            # 2. If no element could be added, try swaps
            if not added:
                swapped = False
                # Try swapping x in S with y not in S
                # We only do this if we are stuck and want to see if we can reach a better state
                # Note: Swaps don't increase size, they just change the set.
                # We should only swap if it allows us to add more elements later.
                # This is complex. 
                # Alternative: Just try all pairs (x, y) and if the new set is dissociated, 
                # check if we can then add *another* element.
                
                # To keep it simple and fast, let's just try to add elements.
                # If we are stuck, this specific init failed to grow further.
                # We can try a "randomized" swap phase here.
                
                # Let's do a simple random swap search for a few iterations
                import random
                random.seed(42)
                for _ in range(100):
                    if not S: break
                    x = random.choice(list(S))
                    y = random.choice([c for c in all_candidates if c not in S])
                    new_S = (S - {x}) | {y}
                    if _is_dissociated(new_S):
                        # Check if this swap allows us to add more
                        can_add = False
                        for z in all_candidates:
                            if z in new_S: continue
                            if _is_dissociated(new_S | {z}):
                                can_add = True
                                break
                        if can_add:
                            S = new_S
                            improved = True
                            swapped = True
                            break
                
                if swapped:
                    continue # Restart the while loop to try adding again
            
        if len(S) > max_size:
            max_size = len(S)
            best_set = S.copy()
            
    # If best_set is empty (shouldn't happen), fallback
    if not best_set:
        return [1]
        
    return list(best_set)

# EVOLVE-BLOCK-END
