"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    def try_greedy(order):
        """
        Greedily build a set from 'order' using incremental sum tracking.
        """
        current_set = []
        L = 1
        sums = {0}  # Set of current subset sums (scaled by L)
        
        for x in order:
            if x in current_set:
                continue
            new_L = lcm(L, x)
            
            # Scale existing sums
            factor = new_L // L
            new_sums = {s * factor for s in sums}
            
            # New weight
            w = new_L // x
            
            # Check for collisions
            valid = True
            # Optimization: if w is large, we can break early? No, just check.
            for s in new_sums:
                if (s + w) in new_sums:
                    valid = False
                    break
            
            if valid:
                # Add new sums
                # Note: new_sums is the set of OLD sums scaled.
                # The new sums are {s + w for s in new_sums}.
                # We update sums to be the union.
                sums = new_sums | {s + w for s in new_sums}
                L = new_L
                current_set.append(x)
                
        return current_set

    best = []
    
    # 1. Deterministic orderings
    base_orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
        # Powers of 2 first
        [1, 2, 4, 8, 16] + [x for x in range(1, 23) if x not in [1, 2, 4, 8, 16]],
        # Primes first
        [2, 3, 5, 7, 11, 13, 17, 19] + [x for x in range(1, 23) if x not in [2, 3, 5, 7, 11, 13, 17, 19]],
        # Large numbers first
        list(range(22, 0, -1)),
    ]
    
    for order in base_orders:
        if time.time() > deadline:
            break
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # 2. Randomized greedy
    random.seed(42)
    powers_of_2 = [1, 2, 4, 8, 16]
    
    for _ in range(2000):
        if time.time() > deadline:
            break
        
        # Generate a random permutation
        # Bias: put powers of 2 in random positions?
        # Or just full random.
        order = list(range(1, 23))
        random.shuffle(order)
        
        # Sometimes try powers of 2 first
        if random.random() < 0.2:
            rest = [x for x in order if x not in powers_of_2]
            order = powers_of_2 + rest
            
        res = try_greedy(order)
        if len(res) > len(best):
            best = res
    
    # 3. Local Search: Try to improve best set
    if len(best) >= 1:
        best_set = set(best)
        # Try to add an element if we remove one
        # For each element in best, remove it, then try to add all others
        # This is expensive, so only do it if time permits and size < 18
        
        if time.time() < deadline - 10.0:
            improved = False
            for x in list(best):
                if time.time() > deadline - 5.0:
                    break
                candidate_set = [y for y in best if y != x]
                candidate_set_set = set(candidate_set)
                
                # Try to add elements from 1..22 not in candidate_set
                # Use greedy on the remaining elements
                remaining = [y for y in range(1, 23) if y not in candidate_set_set]
                # Sort remaining by some heuristic? Random?
                random.shuffle(remaining)
                
                # Greedy add
                current = candidate_set
                L = 1
                sums = {0}
                for y in current:
                    new_L = lcm(L, y)
                    factor = new_L // L
                    new_sums = {s * factor for s in sums}
                    w = new_L // y
                    # Assume valid (it was part of a valid set)
                    sums = new_sums | {s + w for s in new_sums}
                    L = new_L
                    
                for y in remaining:
                    if y in current:
                        continue
                    new_L = lcm(L, y)
                    factor = new_L // L
                    new_sums = {s * factor for s in sums}
                    w = new_L // y
                    valid = True
                    for s in new_sums:
                        if (s + w) in new_sums:
                            valid = False
                            break
                    if valid:
                        sums = new_sums | {s + w for s in new_sums}
                        L = new_L
                        current.append(y)
                        
                if len(current) > len(best):
                    best = current
                    improved = True
                    break # Found improvement, restart local search? Or just break.
            
    if best:
        return best
    
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
