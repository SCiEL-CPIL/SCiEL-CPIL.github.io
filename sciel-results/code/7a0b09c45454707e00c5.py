import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = 512
    
    def has_4ap(S, x):
        """Check if adding x to set S creates a 4-term AP."""
        s = set(S)
        # x can be in position 0, 1, 2, or 3 of a 4-AP
        # a, a+d, a+2d, a+3d where x is one of these
        # x = a: need a+d, a+2d, a+3d in S
        # x = a+d: need a, a+2d, a+3d in S
        # x = a+2d: need a, a+d, a+3d in S
        # x = a+3d: need a, a+d, a+2d in S
        for d in range(1, n+1):
            # x as first term: a=x, check x+d, x+2d, x+3d
            if x + d <= n and x + 2*d <= n and x + 3*d <= n:
                if (x+d) in s and (x+2*d) in s and (x+3*d) in s:
                    return True
            # x as second term: a+d=x, so a=x-d, check a, a+2d, a+3d
            if x - d >= 1 and x + d <= n and x + 2*d <= n:
                if (x-d) in s and (x+d) in s and (x+2*d) in s:
                    return True
            # x as third term: a+2d=x, so a=x-2d, check a, a+d, a+3d
            if x - 2*d >= 1 and x - d >= 1 and x + d <= n:
                if (x-2*d) in s and (x-d) in s and (x+d) in s:
                    return True
            # x as fourth term: a+3d=x, so a=x-3d, check a, a+d, a+2d
            if x - 3*d >= 1 and x - 2*d >= 1 and x - d >= 1:
                if (x-3*d) in s and (x-2*d) in s and (x-d) in s:
                    return True
        return False
    
    def count_4aps(S):
        """Count number of 4-term APs in set S."""
        s = set(S)
        count = 0
        for a in s:
            for d in range(1, n+1):
                if a+d in s and a+2*d in s and a+3*d in s:
                    count += 1
        return count
    
    # Strategy 1: Greedy with multiple orderings
    best_set = set()
    
    # Try different greedy orderings
    orderings = [
        list(range(1, n+1)),
        list(range(n, 0, -1)),
        list(range(1, n+1, 2)),  # odds first
        list(range(2, n+1, 2)),  # evens first
    ]
    
    # Add some random orderings
    for _ in range(10):
        orderings.append(list(range(1, n+1)))
        random.shuffle(orderings[-1])
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = set()
        for x in order:
            if not has_4ap(current, x):
                current.add(x)
        if len(current) > len(best_set):
            best_set = set(current)
    
    # Strategy 2: Local search - try to add elements, remove if conflicts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_set)
        
        # Try adding each element not in current
        for x in range(1, n+1):
            if x in current:
                continue
            if not has_4ap(current, x):
                current.add(x)
                improved = True
        best_set = set(current)
    
    # Strategy 3: Local search with swaps
    # Try swapping one element out and two in, or similar moves
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current = set(best_set)
        best_in_iteration = set(current)
        
        # Try single removals and refills
        for x in list(current):
            candidate = set(current)
            candidate.remove(x)
            # Try to add elements
            for y in range(1, n+1):
                if y not in candidate and not has_4ap(candidate, y):
                    candidate.add(y)
            if len(candidate) > len(best_in_iteration):
                best_in_iteration = set(candidate)
        
        if len(best_in_iteration) > len(best_set):
            best_set = set(best_in_iteration)
        elif iteration > 50:
            # No improvement for a while, try random restarts with perturbation
            perturbed = set(best_set)
            # Remove a random few elements
            to_remove = random.sample(list(perturbed), min(5, len(perturbed)))
            for x in to_remove:
                perturbed.remove(x)
            # Refill greedily
            for y in range(1, n+1):
                if y not in perturbed and not has_4ap(perturbed, y):
                    perturbed.add(y)
            if len(perturbed) > len(best_set):
                best_set = set(perturbed)
    
    # Strategy 4: More aggressive local search - try to find better orderings
    # Use the current best set as a guide
    if time.time() < deadline:
        # Try to build from scratch but biased toward elements in best_set
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Create an ordering that prioritizes elements in best_set
            in_best = [x for x in range(1, n+1) if x in best_set]
            not_in_best = [x for x in range(1, n+1) if x not in best_set]
            random.shuffle(not_in_best)
            order = in_best + not_in_best
            
            current = set()
            for x in order:
                if not has_4ap(current, x):
                    current.add(x)
            if len(current) > len(best_set):
                best_set = set(current)
    
    # Final local search pass
    if time.time() < deadline:
        current = set(best_set)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, n+1):
                if x not in current and not has_4ap(current, x):
                    current.add(x)
                    changed = True
        if len(current) > len(best_set):
            best_set = set(current)
    
    return sorted(best_set)
