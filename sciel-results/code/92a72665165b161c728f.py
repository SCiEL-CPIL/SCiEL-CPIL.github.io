import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # 5s margin
    
    selected = []
    
    # Helper to check if x can be added
    def can_add(x):
        # Check 1: x divides sum of 3 in selected
        # Optimization: Precompute sums? No, just loop.
        for a, b, c in combinations(selected, 3):
            if (a + b + c) % x == 0:
                return False
        
        # Check 2: a in selected divides x + b + c
        for a in selected:
            # We need distinct b, c in selected \ {a}
            # such that a | (x + b + c)
            # Equivalent to: (x + b + c) % a == 0
            
            # To speed up, we can precompute pair sums for each a?
            # But selected changes.
            # For now, simple loop.
            for b, c in combinations(selected, 2):
                if b == a or c == a:
                    continue
                if (x + b + c) % a == 0:
                    return False
        return True

    for x in range(192, 0, -1):
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        
        if can_add(x):
            selected.append(x)
            
    return selected
