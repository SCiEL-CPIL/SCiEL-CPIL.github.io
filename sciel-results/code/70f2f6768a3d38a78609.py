"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85
    
    universe = list(range(1, 23))
    order = sorted(universe, reverse=True)  # Larger elements first
    n = len(order)
    
    best_set = [1]
    best_size = 1
    timed_out = [False]
    
    def backtrack(pos, current_set, sums):
        nonlocal best_set, best_size, timed_out
        if timed_out or time.time() > deadline:
            timed_out = True
            return
        
        # Upper bound: can't exceed current + remaining
        if len(current_set) + (n - pos) <= best_size:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        if pos == n:
            return
        
        elem = order[pos]
        v = 1.0 / elem
        
        # Try including
        sums_set = set(sums)
        valid = True
        for s in sums:
            if s + v in sums_set:
                valid = False
                break
        
        if valid:
            new_sums = list(sums) + [s + v for s in sums]
            current_set.append(elem)
            backtrack(pos + 1, current_set, new_sums)
            current_set.pop()
        
        # Try excluding
        backtrack(pos + 1, current_set, sums)
    
    backtrack(0, [], [0.0])
    
    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
