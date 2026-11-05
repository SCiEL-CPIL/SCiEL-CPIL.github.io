import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Known good solution: {66, ..., 100}, size 35
    best_set = set(range(66, 101))
    best_size = 35
    
    def is_valid(s):
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True

    # Greedy construction starting from large numbers
    # Try to add numbers from 100 down to 1
    # Maintain reachable sums
    current_set = set()
    reachable = {0}
    
    # Order: Try adding large numbers first, then small?
    # Or just iterate 100..1 and add if valid.
    # This yields the interval solution.
    
    # Let's try a different order: Randomized restarts?
    # Given the small N, we can try a few random permutations.
    
    # Strategy:
    # 1. Try the standard interval solution.
    # 2. Try to improve by local search:
    #    - Pick a random element in S, remove it.
    #    - Try to add two elements not in S.
    #    - If size increases and valid, keep.
    
    # Since we are limited to 90s, and N=100 is small, we can do a simple local search.
    
    # Initialize with the best known set
    s = set(best_set)
    
    # Local Search
    # Try to swap elements to increase size
    # This is complex to implement robustly in a short time.
    
    # Alternative:
    # Try all subsets of size 36? Too many.
    
    # Let's stick to the greedy "add if possible" but with different orders.
    # Order 1: 100..1
    # Order 2: 1..100
    # Order 3: Random
    
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1))
    ]
    
    # Add a few random orders
    for _ in range(10):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        s = set()
        reachable = {0}
        
        for x in order:
            # Check if adding x is valid
            # New sums = {y+x for y in reachable if y+x <= 200}
            # If 200 in New sums, invalid.
            
            # Optimization:
            # We only care if 200 is formed.
            # 200 is formed if there exists y in reachable such that y+x == 200 => y == 200-x.
            # So we just check if (200-x) is in reachable.
            
            if (200 - x) not in reachable:
                s.add(x)
                # Update reachable
                # We only need to track sums <= 200
                # new_reachable = reachable U {y+x for y in reachable if y+x <= 200}
                # But we can just update the set.
                new_sums = {y + x for y in reachable if y + x <= 200}
                reachable |= new_sums
            else:
                # Cannot add x
                pass
        
        if len(s) > best_size:
            if is_valid(s):
                best_size = len(s)
                best_set = s
    
    # If we found a better set, return it.
    # Otherwise return the known good one.
    if is_valid(best_set):
        return list(best_set)
    else:
        # Fallback
        return list(range(66, 101))
