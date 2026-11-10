import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit, leaving 35s for evaluator overhead
    
    best_set = []
    best_size = 0
    
    # Helper to check if adding c to current_set is valid
    def try_add(current_set, current_sums, c):
        added = Fraction(1, c)
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + added)
        if len(new_sums) == 2 * len(current_sums):
            return True, new_sums
        return False, None

    # 1. Initial Greedy (Sorted)
    current_set = []
    current_sums = {Fraction(0, 1)}
    for c in range(1, 23):
        ok, new_sums = try_add(current_set, current_sums, c)
        if ok:
            current_set.append(c)
            current_sums = new_sums
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 2. Randomized Restarts
    # We will run multiple iterations
    # To speed up, we can pre-shuffle a few times or just shuffle inside
    
    while time.time() < deadline:
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {Fraction(0, 1)}
        
        for c in candidates:
            if time.time() > deadline:
                break
            
            ok, new_sums = try_add(current_set, current_sums, c)
            if ok:
                current_set.append(c)
                current_sums = new_sums
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    # If we reach a very high size, we might want to stop early?
                    # But we want to maximize.
        
        # Optional: Local Search on the current best
        # Try to swap elements in best_set to allow adding more
        # This is expensive, so we do it sparingly or if we have time
        # Let's do a simple swap check on best_set if we have time
        if time.time() < deadline - 5: # 5s left for final checks
            # Try to add any element to best_set
            # If we can't, try swaps
            best_sums = {Fraction(0, 1)}
            for x in best_set:
                added = Fraction(1, x)
                new_sums = set(best_sums)
                for s in best_sums:
                    new_sums.add(s + added)
                best_sums = new_sums
            
            # Try to add any y not in best_set
            added_any = False
            for y in range(1, 23):
                if y in best_set:
                    continue
                ok, new_sums = try_add(best_set, best_sums, y)
                if ok:
                    best_set.append(y)
                    best_sums = new_sums
                    best_size = len(best_set)
                    added_any = True
                    break # Try to add another? Or restart?
                    # Let's just break and let the next iteration handle it
                    # Or continue adding?
            
            # If we couldn't add, try swaps
            # This part is complex and slow. Let's skip for now and rely on restarts.
            
    if best_set:
        return best_set
    else:
        return [1]
