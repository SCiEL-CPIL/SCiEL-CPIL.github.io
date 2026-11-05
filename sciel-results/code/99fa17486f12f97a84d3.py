import random
import time
from fractions import Fraction
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # The universe of candidates
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Precompute LCM of 1..22 to convert reciprocals to integers
    # LCM(1..22) is 232792560
    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    # Map each x to its integer reciprocal value scaled by L
    # val[x] = L // x
    val_map = {x: L // x for x in candidates}
    
    orders = [
        sorted(candidates, reverse=True),
        sorted(candidates),
        [p for p in candidates if all(p % d != 0 for d in range(2, int(p**0.5)+1))], # Primes
        [n for n in candidates if n > 10], # Large numbers
        [n for n in candidates if n <= 10], # Small numbers
    ]
    
    # Add some random orders
    for _ in range(5):
        random.shuffle(candidates)
        orders.append(candidates[:])
        
    def try_order(order):
        current_set = []
        sums = {0}
        
        for x in order:
            v = val_map[x]
            # Check for collision
            # If (s + v) is in sums for any s in sums, then collision.
            # This is equivalent to: if there exist s1, s2 in sums such that s1 + v = s2.
            # Or simply: new_sums = {s + v for s in sums}. If new_sums & sums is not empty, collision.
            
            # Optimization: Instead of generating all new sums and checking intersection,
            # we can iterate. But for small sets, generating a set and checking intersection is fast.
            new_sums = set()
            collision = False
            for s in sums:
                ns = s + v
                if ns in sums:
                    collision = True
                    break
                new_sums.add(ns)
            
            if not collision:
                current_set.append(x)
                sums.update(new_sums)
                
        return current_set

    # Run through the prepared orders
    for order in orders:
        s = try_order(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
        if time.time() - start_time > timeout:
            break

    # Random restarts with integer arithmetic is much faster
    while time.time() - start_time < timeout:
        random.shuffle(candidates)
        s = try_order(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            if best_size >= 16:
                 # Keep searching but maybe stop if we hit a very high bar? 
                 # For now, just update best.
                 pass
             
    if not best_set:
        # Fallback to a known valid small set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best_set
# EVOLVE-BLOCK-END
