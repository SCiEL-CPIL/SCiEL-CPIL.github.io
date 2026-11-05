import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 88.0  # Safety margin for 90s kill
    
    # Generate all 4-subsets of {0, ..., 15} as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    best_family = []
    
    # Optimized validity check
    def is_valid(x, f):
        n = len(f)
        for i in range(n):
            a = f[i]
            for j in range(i + 1, n):
                b = f[j]
                k = a & b
                if (x & a == k) and (x & b == k):
                    return False
        return True

    # Randomized Greedy with Restarts
    # We can also try different ordering strategies if time permits
    while time.time() - start_time < time_limit:
        # Shuffle a copy to avoid modifying all_sets if we want to reuse? 
        # Shuffling in place is faster.
        random.shuffle(all_sets)
        family = []
        
        for x in all_sets:
            if is_valid(x, family):
                family.append(x)
        
        if len(family) > len(best_family):
            best_family = family
            # If we reach a very high number, we might stop early?
            # But better to keep searching for the max.
            
    # Convert masks back to sorted lists of 1-based integers
    result = []
    for mask in best_family:
        elements = []
        for i in range(16):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result
