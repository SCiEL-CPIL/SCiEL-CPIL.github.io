import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial solution: {51, ..., 100}
    # This is known to be valid and size 50.
    current_set = set(range(51, 101))
    current_size = len(current_set)
    
    # Helper to check if a set is valid (no subset sums to 200)
    # This is expensive. We need a faster check or incremental update.
    # For local search, we can check the effect of a swap/insert.
    
    # Precompute subset sums for the current set?
    # With size 50, subset sums can be many.
    # However, we only care if 200 is reachable.
    # We can maintain a bitset of reachable sums up to 200.
    
    def get_reachable_sums(S):
        reachable = {0}
        for x in S:
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 in new_sums:
                return None # Invalid
            reachable |= new_sums
        return reachable

    # Verify initial set
    init_reachable = get_reachable_sums(current_set)
    if init_reachable is None:
        # Fallback to greedy
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    best_set = set(current_set)
    best_size = current_size
    
    # Local Search: Try to increase size by 1.
    # Strategy: Remove one element from current_set, add two from [1..50].
    # Or: Add one from [1..50], remove one from current_set if conflict, etc.
    
    # Since we suspect 50 is max, we will likely fail to find 51.
    # But we must try to be sure.
    
    # Optimization: Use bitsets for speed?
    # Python sets of ints are okay for 200.
    
    # Let's try a few random swaps to see if we can get lucky.
    # If we find a valid set of size 51, we return it.
    
    candidates_small = list(range(1, 51))
    candidates_large = list(range(51, 101))
    
    # To save time, we will only do a limited number of iterations.
    max_iters = 5000
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Randomly pick a large element to remove
        L = random.choice(candidates_large)
        if L not in current_set:
            continue
            
        # Randomly pick two small elements to add
        # Ensure they are not already in set (they aren't, set is 51-100)
        # But we need distinct x, y
        x, y = random.sample(candidates_small, 2)
        
        # Proposed new set
        new_set = set(current_set)
        new_set.remove(L)
        new_set.add(x)
        new_set.add(y)
        
        # Check validity
        # Instead of full recompute, we can check if 200 is formed.
        # But incremental check is complex. Full recompute is O(N * Sums).
        # Sums <= 200. N=51. 51*200 = 10200 ops. Very fast.
        
        reachable = {0}
        valid = True
        for z in new_set:
            new_sums = {s + z for s in reachable if s + z <= 200}
            if 200 in new_sums:
                valid = False
                break
            reachable |= new_sums
            
        if valid:
            return list(new_set) # Found size 51!
            
    # If no improvement found, return best set (size 50)
    return list(best_set)
