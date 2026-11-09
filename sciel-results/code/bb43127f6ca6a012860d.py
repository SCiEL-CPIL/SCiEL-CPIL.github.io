import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    # Initial set: all large numbers 65..128
    # This is a safe base.
    S_base = set(range(65, 129))
    
    def is_valid_addition(c, S_current):
        # S_current includes c? No, we check before adding.
        # c is NOT in S_current.
        for x in S_current:
            for y in S_current:
                if x >= y:
                    continue
                # If c+x and c+y are in S_current, and x+y is in S_current or x+y==c
                # Then (c,x,y) is a forbidden triple in S_current U {c}
                if (c + x in S_current) and (c + y in S_current):
                    if (x + y in S_current) or (x + y == c):
                        return False
        return True

    def greedy(order):
        S = set(S_base)
        for c in order:
            if is_valid_addition(c, S):
                S.add(c)
        return S

    best_set = None
    best_size = -1
    
    # Try a few deterministic orders
    orders = [
        list(range(1, 65)),
        list(range(64, 0, -1)),
        # Maybe even numbers first?
        [i for i in range(1, 65) if i % 2 == 0] + [i for i in range(1, 65) if i % 2 == 1],
    ]
    
    for order in orders:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts
    for _ in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, 65))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local Search: Hill Climbing
    # Try to add any valid number
    current_set = best_set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(1, 65):
            if c not in current_set:
                if is_valid_addition(c, current_set):
                    current_set.add(c)
                    improved = True
    
    # Try to swap: remove one, add another
    # This is harder. We can try to find a c_remove such that removing it allows adding k > 1 new numbers.
    # Or just swap 1 for 1 if it leads to a better state?
    # Given the "add all possible" loop, we are at a local maximum for additions.
    # Swaps might help escape local optima.
    
    # Simple Swap Search
    for _ in range(50):
        if time.time() > deadline:
            break
        # Pick a random element to remove
        if not current_set:
            break
        # Only remove small elements? Large elements are "free".
        # Removing a large element doesn't help add small ones easily?
        # Actually, removing a large element might allow a small element if the small element's sums with other smalls were blocked by the large element's presence?
        # No, the large element was in the set, so its sums were "available".
        # Removing it makes sums "unavailable", which makes it HARDER to add new elements.
        # So removing large elements is bad.
        # We should only consider removing small elements.
        
        smalls = [x for x in current_set if x <= 64]
        if not smalls:
            break
            
        c_remove = random.choice(smalls)
        temp_set = current_set - {c_remove}
        
        # Now try to add as many as possible
        # We can run the "add all possible" loop on temp_set
        improved_temp = True
        while improved_temp:
            improved_temp = False
            for c in range(1, 65):
                if c not in temp_set:
                    if is_valid_addition(c, temp_set):
                        temp_set.add(c)
                        improved_temp = True
                        
        if len(temp_set) > len(current_set):
            current_set = temp_set
            best_size = len(current_set)
            
    # Final polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in range(1, 65):
            if c not in current_set:
                if is_valid_addition(c, current_set):
                    current_set.add(c)
                    improved = True

    result = sorted(list(current_set))
    return result
