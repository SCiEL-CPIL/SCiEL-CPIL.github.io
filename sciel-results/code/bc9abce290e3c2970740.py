import time
import random

N = 512

# Precompute forbidden masks for bit-parallel conflict checking
# mask[i] is a bitmask where bit j is set if i and j are part of the same 4-AP
# We need to check if adding x to S creates a 4-AP.
# A 4-AP is (a, b, c, d). If x is in the AP, the other 3 must be in S.
# Instead of checking "others in S", we can precompute for each x, the set of y such that
# x and y are in a 4-AP. But that's not enough. We need to know if the *other two* are also in S.
#
# Let's use a different bit-parallel approach.
# Let S be a bitmask.
# For a candidate x, x can be added if there is no 4-AP (a,b,c,d) containing x such that the other 3 are in S.
# This is hard to do with a single bitmask AND.
#
# Alternative: Precompute for each x, a list of "forbidden triples" (y, z, w) such that {x, y, z, w} is a 4-AP.
# This is what the original code did with aps_by_element.
# The bottleneck was the `in S` checks.
#
# Let's try to optimize the check using bitmasks.
# Let S_mask be an integer representing the set S.
# For a candidate x, we want to check if any 4-AP containing x is fully contained in S ∪ {x}.
# This is equivalent to checking if there exists a 4-AP (a,b,c,d) with x in it, and the other 3 in S.
#
# Let's precompute for each x, a list of masks.
# For each 4-AP containing x, create a mask of the other 3 elements.
# Then check if (S_mask & mask) == mask.
#
# This reduces the check to bitwise AND and comparison.
# The number of 4-APs containing x is small (O(N)).
# So for each x, we have a list of ~O(N) masks.
# Checking all of them is O(N) bitwise ops.
# This is likely faster than O(N) set lookups.

# Precompute masks for each element
# masks[x] is a list of integers. Each integer is a bitmask of the other 3 elements in a 4-AP with x.
masks = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # AP: (a, b, c, e)
        # For a: others are b, c, e
        mask_a = (1 << b) | (1 << c) | (1 << e)
        masks[a].append(mask_a)
        
        # For b: others are a, c, e
        mask_b = (1 << a) | (1 << c) | (1 << e)
        masks[b].append(mask_b)
        
        # For c: others are a, b, e
        mask_c = (1 << a) | (1 << b) | (1 << e)
        masks[c].append(mask_c)
        
        # For e: others are a, b, c
        mask_e = (1 << a) | (1 << b) | (1 << c)
        masks[e].append(mask_e)

def can_add_mask(x, S_mask):
    """
    Check if x can be added to set represented by S_mask without creating a 4-AP.
    Returns True if safe.
    """
    for m in masks[x]:
        if (S_mask & m) == m:
            return False
    return True

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50.0
    
    best_set = set()
    best_size = 0
    best_mask = 0
    
    # We will run multiple restarts with random orderings
    # Each restart:
    # 1. Shuffle [1, N]
    # 2. Greedily add elements if they don't create a 4-AP
    # 3. Do a second pass to add any remaining elements that can now be added
    
    while time.time() < deadline:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S_mask = 0
        
        # First pass: Greedy
        for x in candidates:
            if can_add_mask(x, S_mask):
                S_mask |= (1 << x)
        
        # Second pass: Try to add any elements that were skipped
        for x in range(1, N + 1):
            if not (S_mask & (1 << x)):
                if can_add_mask(x, S_mask):
                    S_mask |= (1 << x)
        
        # Convert mask to set for size calculation and storage
        # We can count bits directly
        current_size = bin(S_mask).count('1')
        
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            
    # Convert best_mask to sorted list
    best_set = [i for i in range(1, N + 1) if (best_mask & (1 << i))]
    return best_set
