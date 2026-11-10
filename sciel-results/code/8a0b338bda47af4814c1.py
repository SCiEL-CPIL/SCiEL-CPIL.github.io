import time
import random

N = 256

def construct():
    # Deadline guard: 30s limit, use 25s to be safe
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_set = None
    best_size = float('inf')
    
    # Helper to check if x can be added to current_set with existing sums
    def can_add(x, current_set, sums):
        new_sums = [x + a for a in current_set]
        new_sums.append(2 * x)
        
        # Check internal distinctness of new sums
        if len(set(new_sums)) != len(new_sums):
            return False, None
            
        # Check against existing sums
        if sums.intersection(new_sums):
            return False, None
            
        return True, new_sums

    # Run multiple random greedy attempts
    while time.time() < deadline:
        # Random order of candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        current_set = []
        sums = set()
        
        for x in candidates:
            ok, new_sums = can_add(x, current_set, sums)
            if ok:
                current_set.append(x)
                sums.update(new_sums)
                
        # The resulting set is maximal Sidon
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
            # Optional: If we find a very small set, we might break early?
            # Theoretical lower bound is roughly sqrt(N) ~ 16. 
            # Maximal sets can be smaller than minimum Sidon sets? 
            # No, a maximal Sidon set is a Sidon set, so its size is at least the minimum Sidon set size.
            # Min Sidon set size for N=256 is around 16-20.
            # If we hit 16 or 17, it's likely near optimal.
            if best_size <= 17:
                break

    if best_set is None:
        # Fallback to deterministic if random failed or time was too short for even one
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    return best_set
