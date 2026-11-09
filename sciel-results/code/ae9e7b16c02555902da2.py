"""
Optimized search for maximum-cardinality reciprocal-dissociated subset of [1..22].
Uses multiple greedy constructions with random orderings.
"""
import time
import random

# LCM of 1..22
_LCM = 232792560

def _verify_dissociated(subset):
    """
    Return True if all reciprocal subset sums are distinct.
    Uses integer arithmetic with LCM.
    """
    if not subset:
        return True
    
    values = [_LCM // i for i in subset]
    
    # Maintain a set of all possible subset sums of the current prefix
    sums = {0}
    
    for v in values:
        # If we add v, the new sums are {s + v for s in sums}
        # A collision occurs if any (s + v) is already in sums.
        # This is equivalent to checking if (sums + v) intersects sums.
        # Since v > 0, we can just check.
        # Optimization: if len(sums) * 2 > some limit, it's still fine for k<=20.
        
        # Check for collision
        for s in sums:
            if (s + v) in sums:
                return False
        
        # Update sums
        # To avoid modifying the set while iterating (though we iterated over the old view),
        # we create a list of new sums.
        new_sums = [s + v for s in sums]
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Strict safety margin
    
    all_elements = list(range(1, 23))
    best_set = [1]  # Fallback
    best_size = 1
    
    random.seed(42)
    
    # We will run multiple greedy attempts.
    # Each attempt picks a random permutation and greedily adds elements.
    # To diversify, we can also try sorting by reciprocal value (descending/ascending) occasionally.
    
    while time.time() < deadline:
        # Generate a random permutation
        perm = all_elements[:]
        random.shuffle(perm)
        
        # Occasionally try sorted orders which might find specific structures
        if random.random() < 0.1:
            perm.sort(reverse=True) # Large numbers first (safer)
        elif random.random() < 0.1:
            perm.sort() # Small numbers first (risky, but might work if lucky)
        
        current_set = []
        current_sums = {0} # We can optimize by keeping sums, but _verify_dissociated is simple.
                           # Let's try to keep the incremental check for speed.
        
        # Incremental greedy construction
        valid = True
        for i in perm:
            if time.time() > deadline:
                break
            
            v = _LCM // i
            
            # Check if adding v causes a collision with current_sums
            # Collision if (s + v) in current_sums for any s in current_sums
            collision = False
            for s in current_sums:
                if (s + v) in current_sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(i)
                # Update current_sums
                new_sums = [s + v for s in current_sums]
                current_sums.update(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        # If we found a set of size 22, we are done (max possible)
        if best_size == 22:
            break

    return best_set
