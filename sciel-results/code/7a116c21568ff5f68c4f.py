import time
import random

# Precompute conflicts for 4-APs
# A 4-AP is a, a+d, a+2d, a+3d
# For each x, we store the other 3 elements for every 4-AP x is part of.
# x can be at position 1, 2, 3, or 4 in the AP.

def construct():
    n = 512
    # conflicts[x] is a list of tuples (y, z, w)
    # If x, y, z, w form a 4-AP, then adding x is forbidden if y,z,w are in S.
    conflicts = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs
    for d in range(1, (n // 3) + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # The AP is (a, b, c, e)
            # For a: others are (b, c, e)
            conflicts[a].append((b, c, e))
            # For b: others are (a, c, e)
            conflicts[b].append((a, c, e))
            # For c: others are (a, b, e)
            conflicts[c].append((a, b, e))
            # For e: others are (a, b, c)
            conflicts[e].append((a, b, c))

    def is_valid_addition(x, in_set):
        for y, z, w in conflicts[x]:
            if in_set[y] and in_set[z] and in_set[w]:
                return False
        return True

    best_set = set()
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    while time.time() < deadline:
        in_set = [False] * (n + 1)
        current_size = 0
        
        # Randomized Greedy Construction
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(x, in_set):
                in_set[x] = True
                current_size += 1
        
        # Local Improvement: Try to add any element that was skipped
        # This handles cases where order mattered
        for x in range(1, n + 1):
            if not in_set[x] and is_valid_addition(x, in_set):
                in_set[x] = True
                current_size += 1
                
        # Record best
        if current_size > best_size:
            best_size = current_size
            best_set = set(i for i in range(1, n + 1) if in_set[i])
            
        # Optional: Try a few swaps or perturbations if time permits
        # For now, rely on many restarts
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
