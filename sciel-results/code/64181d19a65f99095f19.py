import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _is_valid(subset):
    """
    Check if all subset sums of reciprocals are distinct.
    Equivalent to checking if all subset sums of LCM/i are distinct.
    """
    if not subset:
        return True
    
    # Convert to integer weights
    weights = [_LCM // i for i in subset]
    k = len(weights)
    
    # Generate all subset sums
    # For k up to 22, 2^k can be large, but we expect k <= 18 or so.
    # 2^18 = 262,144 which is fine.
    # We use a set to store sums.
    sums = {0}
    for w in weights:
        # Add w to all existing sums
        new_sums = {s + w for s in sums}
        if new_sums & sums:
            return False
        sums |= new_sums
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19] # Seed
    best_size = len(best_subset)
    
    # Precompute weights
    weights = {_LCM // i for i in range(1, 23)}
    
    # Strategy 1: Greedy with random restarts
    # Try to build a set by adding elements in random order
    for _ in range(2000):
        if time.time() > deadline:
            break
            
        # Random permutation of 1..22
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current = []
        for c in candidates:
            # Check if adding c is valid
            # Optimization: only check if current is valid (it is by construction)
            # and if current + [c] is valid
            if _is_valid(current + [c]):
                current.append(c)
                
        if len(current) > best_size:
            best_size = len(current)
            best_subset = current
            # If we found 18 or more, we can stop or continue to see if 19 is possible
            if best_size >= 18:
                 # Continue searching for larger, but maybe slow down
                 pass

    # Strategy 2: Local search / Hill climbing
    # Start from best_subset and try to swap elements or add/remove
    # We can try to add elements to best_subset
    # Or start with a large random set and remove invalid elements
    
    # Let's try to find a set of size 16+ more aggressively
    # Random restarts with repair
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Start with a random set of size 15
        # Or start with all 22 and remove
        # Removing is expensive if we check validity each time.
        # Let's try: Start with random 16 elements.
        # If valid, try to add more.
        # If not valid, try to swap one element with another to fix.
        
        # Simpler: Random subset of size 15
        # Shuffle 1..22, take first 15
        perm = list(range(1, 23))
        random.shuffle(perm)
        current = perm[:15]
        
        if _is_valid(current):
            # Try to add more
            remaining = [x for x in range(1, 23) if x not in current]
            for c in remaining:
                if _is_valid(current + [c]):
                    current.append(c)
            
            if len(current) > best_size:
                best_size = len(current)
                best_subset = current
        else:
            # Try to repair by swapping
            # This is complex. Let's just rely on the greedy restarts which are effective.
            pass

    return best_subset
