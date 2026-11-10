"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce

def _is_dissociated(elems):
    """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
    if not elems:
        return True
    if len(elems) == 1:
        return True
    
    n = len(elems)
    # Calculate LCM
    L = reduce(math.lcm, elems)
    
    # Convert to scaled integers
    weights = [L // e for e in elems]
    
    sums = {0}
    for w in weights:
        # Check for collision: does s + w exist in sums for any s in sums?
        # Equivalent to: (sums + w) & sums is not empty
        # Early termination:
        for s in sums:
            if s + w in sums:
                return False
        # Update sums
        # We can do this efficiently
        # new_sums = {s + w for s in sums}
        # sums = sums | new_sums
        # To save time, we can just update in place if we are careful, 
        # but sets are immutable-ish in terms of size during iteration.
        # So we create a new set.
        sums = sums | {s + w for s in sums}
        
    return len(sums) == (1 << n)

def construct():
    start_time = time.time()
    deadline = start_time + 55 # 55s limit, keep 5s buffer for safety/overhead
    
    universe = list(range(1, 23))
    
    # Greedy function with a given order
    def greedy(order):
        cur = []
        for e in order:
            test_set = cur + [e]
            if _is_dissociated(test_set):
                cur = test_set
        return cur

    best_set = []
    
    # 1. Try deterministic orders
    orders = [
        sorted(universe, reverse=True),
        sorted(universe),
        list(universe)
    ]
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        result = greedy(order)
        if len(result) > len(best_set):
            best_set = result
    
    # 2. Local search: try to add any missing element.
    def local_search_add(s):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            remaining = [e for e in universe if e not in s]
            random.shuffle(remaining)
            for e in remaining:
                test_set = s + [e]
                if _is_dissociated(test_set):
                    s = test_set
                    improved = True
                    break
        return s

    best_set = local_search_add(best_set)

    # 3. Random restarts with local search
    num_restarts = 0
    while time.time() < deadline - 15:
        random.shuffle(universe)
        result = greedy(universe)
        result = local_search_add(result)
        
        if len(result) > len(best_set):
            best_set = result
            # If we found a very large set, we might stop early?
            if len(best_set) >= 20:
                break
        num_restarts += 1
        
    # 4. Local search with swaps to escape local maxima
    # Try to swap an element in best_set with one outside to see if it allows growth
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        s_set = set(best_set)
        in_set = list(best_set)
        out_set = [e for e in universe if e not in s_set]
        random.shuffle(out_set)
        
        for u in in_set:
            if time.time() > deadline - 5:
                break
            for v in out_set:
                if time.time() > deadline - 5:
                    break
                # Try swap
                new_s = [x for x in best_set if x != u] + [v]
                if _is_dissociated(new_s):
                    # Check if this swap allows adding more
                    extended = local_search_add(new_s)
                    if len(extended) > len(best_set):
                        best_set = extended
                        improved = True
                        break
            if improved:
                break
    
    # 5. Kick mechanism: Random 2-swap to escape deep local optima
    # If we are stuck at size K, try removing 2 and adding 2
    # This is expensive, so we do it sparingly
    if time.time() < deadline - 10:
        s_set = set(best_set)
        in_set = list(best_set)
        out_set = [e for e in universe if e not in s_set]
        
        # Try a few random 2-swaps
        for _ in range(20): # Limited attempts
            if time.time() > deadline - 5:
                break
            if len(in_set) < 2 or len(out_set) < 2:
                break
            
            u1, u2 = random.sample(in_set, 2)
            v1, v2 = random.sample(out_set, 2)
            
            new_s = [x for x in best_set if x not in (u1, u2)] + [v1, v2]
            if _is_dissociated(new_s):
                extended = local_search_add(new_s)
                if len(extended) > len(best_set):
                    best_set = extended
                    # Update in_set and out_set
                    s_set = set(best_set)
                    in_set = list(best_set)
                    out_set = [e for e in universe if e not in s_set]

    # Final verification
    if best_set and _is_dissociated(best_set):
        return sorted(best_set)
    
    # Fallback to a known valid set if something went wrong
    # Size 9 was the seed. Let's return a size 9 set.
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
