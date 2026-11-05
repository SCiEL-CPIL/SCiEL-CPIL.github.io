import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that lcm(i, j) <= 256
    # We only care about i, j in [2, 256]
    N = 256
    conflicts = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Initial solution: all numbers in [129, 256]
    # This is valid as proven above.
    best_set = set(range(129, N + 1))
    best_score = sum(1.0 / x for x in best_set)
    
    def is_compatible(x, current_set):
        # Check if x can be added to current_set
        # current_set must not contain any element from conflicts[x]
        for y in conflicts[x]:
            if y in current_set:
                return False
        return True

    def score_diff(x_add, y_remove):
        return 1.0 / x_add - 1.0 / y_remove

    while time.time() < deadline:
        improved = False
        
        # 1. Try to add a single element
        best_add = None
        best_add_val = 0.0
        for x in range(2, N + 1):
            if x not in best_set:
                if is_compatible(x, best_set):
                    val = 1.0 / x
                    if val > best_add_val:
                        best_add_val = val
                        best_add = x
        
        if best_add is not None:
            best_set.add(best_add)
            best_score += best_add_val
            improved = True
            continue # Restart search after modification
        
        # 2. Try to swap one element for another (remove y, add x)
        best_swap = None
        best_swap_gain = 0.0
        best_swap_x = None
        best_swap_y = None
        
        # Optimization: Only consider x < y for potential gain
        for y in list(best_set):
            for x in range(2, y):
                if x in best_set:
                    continue
                # Check if x is compatible with (best_set - {y})
                # This means conflicts[x] should not intersect (best_set - {y})
                # i.e., conflicts[x] & best_set should be a subset of {y}
                # If conflicts[x] contains any element from best_set other than y, it's invalid.
                # Also, x must not be in best_set (already checked)
                
                # Efficient check:
                valid = True
                for z in conflicts[x]:
                    if z in best_set and z != y:
                        valid = False
                        break
                if valid:
                    gain = 1.0 / x - 1.0 / y
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_x = x
                        best_swap_y = y
        
        if best_swap_x is not None:
            best_set.remove(best_swap_y)
            best_set.add(best_swap_x)
            best_score += best_swap_gain
            improved = True
            continue # Restart search after modification
            
        if not improved:
            # Local optimum reached. 
            # To escape, we could do a random restart or perturbation.
            # For now, break to save time, or try a simple perturbation.
            # Given the time, let's try a few random restarts with different initial sets.
            # But the current set is likely very good.
            # Let's break to return the best found so far.
            break
            
    return list(best_set)
