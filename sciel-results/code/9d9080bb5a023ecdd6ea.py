"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_size = 0
    best_set = []
    
    # Helper to check if a set has unique triple sums
    def is_valid(s):
        if not s:
            return True
        s = sorted(s)
        n = len(s)
        used = set()
        # Triples (i, j, k) with i <= j <= k
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    val = s[i] + s[j] + s[k]
                    if val in used:
                        return False
                    used.add(val)
        return True

    # Helper to find a specific collision
    def find_collision(s):
        if not s:
            return None
        s = sorted(s)
        n = len(s)
        sums = {}
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    val = s[i] + s[j] + s[k]
                    if val in sums:
                        # Return indices of the two triples that collide
                        return (sums[val], (i, j, k))
                    sums[val] = (i, j, k)
        return None

    # Try to improve a set by swapping out bad elements
    def improve(s, max_iters=50):
        s = sorted(s)
        if not is_valid(s):
            return s
            
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Try to add a random new element
            # We want to maximize size, so we try to find an element that fits
            # But since we start with a valid set, we might try to expand?
            # The task is to maximize cardinality. 
            # Strategy: Start with a small valid set, try to grow it.
            # If we can't grow, maybe we can't improve.
            
            # Let's stick to a different approach:
            # 1. Generate a random valid set of size K.
            # 2. Try to add elements to make it size K+1.
            
            pass # This function structure is a bit off for the main loop. Let's restructure.

    def try_grow(current_set, n=4096):
        """Try to add one element to current_set while maintaining validity."""
        if not current_set:
            return current_set
            
        current_set = sorted(current_set)
        
        # Precompute existing sums for speed? 
        # O(N^3) check is expensive. 
        # Let's use the incremental check:
        # When adding x, new sums are x+x+x, x+x+y, x+y+z (where y,z in current_set)
        # We need to check if any of these new sums collide with existing sums or each other.
        
        existing_sums = set()
        n_curr = len(current_set)
        for i in range(n_curr):
            for j in range(i, n_curr):
                for k in range(j, n_curr):
                    existing_sums.add(current_set[i] + current_set[j] + current_set[k])
                    
        # Candidate range: We can pick any number in [1, n] not in current_set.
        # To maximize chance, maybe pick randomly? Or scan?
        # Scanning 1..4096 for each addition is 4096 * (check cost).
        # Check cost is O(N^2) for new sums + O(N^3) for existing? No, existing is precomputed.
        # New sums involving x:
        # 1. x+x+x
        # 2. x+x+y for y in S
        # 3. x+y+z for y<=z in S
        # Total new sums: 1 + N + N(N+1)/2 ~ N^2/2.
        # We need to check if any new sum is in existing_sums OR if any two new sums are equal.
        
        # Optimization: 
        # 1. Check x+x+x against existing_sums.
        # 2. Check x+x+y against existing_sums and other new sums.
        
        # Let's just try a few random candidates to save time, or a structured scan.
        # Given N is small (<20 initially), N^2 is small.
        
        candidates = [x for x in range(1, n+1) if x not in current_set]
        random.shuffle(candidates)
        
        # Try a subset of candidates to avoid timeout
        max_tries = 100
        for x in candidates[:max_tries]:
            new_sums = []
            # Generate new sums
            new_sums.append(x+x+x)
            for y in current_set:
                new_sums.append(x+x+y)
            for i in range(n_curr):
                for j in range(i, n_curr):
                    new_sums.append(x + current_set[i] + current_set[j])
            
            # Check for internal collisions in new_sums
            if len(new_sums) != len(set(new_sums)):
                continue
                
            # Check for collisions with existing_sums
            collision = False
            for s_val in new_sums:
                if s_val in existing_sums:
                    collision = True
                    break
                    
            if not collision:
                return current_set + [x]
                
        return current_set

    # Main Loop: Randomized Growth
    # We try multiple restarts with different random seeds/starting points
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Start with a small random valid set
        # Or start empty and grow?
        # Starting empty and growing is O(K * (N^2 * CheckCost)).
        # CheckCost is O(N^2) for new sums + O(1) lookup.
        # So growth step is O(N^2).
        # Total O(K^3). For K=20, 8000 ops. Very fast.
        
        current = []
        
        # Try to grow to max size
        while True:
            if time.time() > deadline:
                break
                
            next_set = try_grow(current, 4096)
            if len(next_set) == len(current):
                break
            current = next_set
            
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
            
    return best_set
# EVOLVE-BLOCK-END
