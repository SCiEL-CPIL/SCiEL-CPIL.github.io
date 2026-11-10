import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit

    best_set = []
    best_size = 0

    # Precompute LCM of 1..22
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = reduce(lcm, range(1, 23))
    weights = {a: M // a for a in range(1, 23)}
    
    # Helper to check if adding weight w to current_sums is valid
    # current_sums is a set of ints
    def try_add_int(current_sums, w):
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
        if len(new_sums) == 2 * len(current_sums):
            return True, new_sums
        return False, None

    # 1. Initial Greedy (Sorted by weight descending, i.e., a ascending)
    current_set = []
    current_sums = {0}
    for a in range(1, 23):
        w = weights[a]
        ok, new_sums = try_add_int(current_sums, w)
        if ok:
            current_set.append(a)
            current_sums = new_sums
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 2. Randomized Restarts with Integer Arithmetic
    # We will run multiple iterations
    while time.time() < deadline:
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {0}
        
        for a in candidates:
            if time.time() > deadline:
                break
            
            w = weights[a]
            ok, new_sums = try_add_int(current_sums, w)
            if ok:
                current_set.append(a)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
        
        # Optional: Local Search / Swap attempts
        # Try to add any element to best_set if possible
        if time.time() < deadline - 5:
            # Reconstruct sums for best_set
            best_sums = {0}
            for x in best_set:
                w_x = weights[x]
                new_sums = set(best_sums)
                for s in best_sums:
                    new_sums.add(s + w_x)
                best_sums = new_sums
            
            # Try to add any y not in best_set
            added_any = False
            for y in range(1, 23):
                if y in best_set:
                    continue
                w_y = weights[y]
                ok, new_sums = try_add_int(best_sums, w_y)
                if ok:
                    best_set.append(y)
                    best_sums = new_sums
                    best_size = len(best_set)
                    added_any = True
                    # Break to restart loop or continue? 
                    # If we added one, we might be able to add another.
                    # Let's continue the loop to try adding more.
                    # But `best_sums` is updated.
                    # We need to be careful with `y in best_set` check if we append.
                    # Since we iterate `range(1, 23)`, and append to `best_set`, 
                    # the `if y in best_set` check will prevent re-adding.
                    pass
            
            # If no element could be added, try swaps?
            # Swaps are complex. For now, rely on restarts.
            # The integer speedup should allow many more restarts.

    if best_set:
        return best_set
    else:
        return [1]
