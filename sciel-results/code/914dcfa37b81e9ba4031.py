"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    n = 255
    
    # Generate all multiples of 3 in [1, 255]
    base_set = [x for x in range(1, 256) if x % 3 == 0]
    base_set_set = set(base_set)
    
    # Try to add elements not in base_set that share a common factor > 1 with all elements in base_set
    # Since all elements in base_set are multiples of 3, any element we add must also be a multiple of 3
    # (otherwise gcd with some element might be 1)
    # Actually, let's think more carefully.
    # If we add an element x that is NOT a multiple of 3, then for all y in base_set, gcd(x,y) > 1 must hold.
    # This means x must share a prime factor with every y in base_set.
    # But base_set contains 3 itself (gcd(x,3) > 1 means x is divisible by 3 or shares another factor... wait 3 is prime)
    # So gcd(x, 3) > 1 means x is divisible by 3.
    # Therefore, any element we add must be divisible by 3.
    # So base_set is already maximal in this sense.
    
    # Let's verify: is base_set maximal?
    # Could we remove some elements and add others to get a larger set?
    # The constraint is: for all distinct pairs (a,b) in the set, gcd(a,b) > 1.
    # And the set must contain 255.
    
    # One approach: find the maximum clique in a graph where vertices are [1,255] and edges connect
    # numbers with gcd > 1. But we also require 255 to be in the set.
    # This is a large graph (255 vertices), but we can try a greedy approach.
    
    # Actually, the set of all multiples of 3 is a valid set.
    # Can we do better by using a different "hub" prime or a more complex structure?
    
    # Let's try a greedy approach: start with 255, then greedily add elements that share a factor with all current elements.
    # This is a heuristic and may not find the optimum, but it's fast.
    
    # Actually, let's think about the structure. If the set contains 255, then every element in the set
    # must share a factor with 255. The factors of 255 are 3, 5, 17.
    # So every element in the set must be divisible by 3, 5, or 17.
    
    # Let S be a valid set containing 255.
    # Let S3 = {x in S : 3 divides x}, S5 = {x in S : 5 divides x}, S17 = {x in S : 17 divides x}.
    # For any x in S3 and y in S5, gcd(x,y) > 1. Since x is divisible by 3 and y is divisible by 5,
    # they must share another factor. This is a strong constraint.
    
    # The simplest valid set is all multiples of 3 (which includes 255). Size = 85.
    # Can we get more than 85?
    
    # Let's try to add some multiples of 5 that are not multiples of 3.
    # If x is a multiple of 5 but not 3, then for all y in S (where S currently contains all multiples of 3),
    # we need gcd(x,y) > 1. In particular, gcd(x,3) > 1, which is false since x is not a multiple of 3.
    # Wait, 3 is in S (since 3 is a multiple of 3). So gcd(x,3) must be > 1, which means x must be a multiple of 3.
    # So we cannot add any element that is not a multiple of 3 if 3 is in the set.
    
    # What if we remove 3 from the set?
    # Let S' = {multiples of 3 in [1,255]} \ {3}.
    # Now can we add some multiples of 5?
    # If x is a multiple of 5 but not 3, we need gcd(x,y) > 1 for all y in S'.
    # S' contains 6, 9, 12, 15, ...
    # gcd(x, 9) > 1: x must share a factor with 9, so x must be a multiple of 3. Contradiction.
    # So we still can't add non-multiples of 3.
    
    # What if we remove all small multiples of 3?
    # This is getting complicated. The greedy approach might help.
    
    # Let's implement a greedy algorithm:
    # 1. Start with {255}.
    # 2. For each candidate in [1, 255] \ {255}, check if it can be added (shares factor with all current elements).
    # 3. If yes, add it.
    # 4. Repeat until no more elements can be added.
    
    # This is O(n^3) in the worst case, but n=255 is small enough.
    
    current = {255}
    candidates = [x for x in range(1, 256) if x != 255]
    
    # Sort candidates by number of common factors with current set (greedy heuristic)
    # Actually, let's just iterate and add if possible.
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # Check if candidate can be added
        can_add = True
        for elem in current:
            if math.gcd(candidate, elem) == 1:
                can_add = False
                break
        
        if can_add:
            current.add(candidate)
    
    # Convert to list
    result_greedy = list(current)
    
    # Compare with base_set
    if len(result_greedy) > len(base_set):
        return result_greedy
    else:
        return base_set

# EVOLVE-BLOCK-END
