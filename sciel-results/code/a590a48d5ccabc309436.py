import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for evaluation overhead
    
    n = 16
    k = 4
    max_size = 250
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(1, n + 1), k):
        mask = 0
        for x in combo:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    total_sets = len(all_sets)
    
    best_family = []
    
    # Try multiple random restarts
    while time.time() < deadline:
        random.shuffle(all_sets)
        family = []
        family_set = set() # For O(1) lookup if needed, but we only add
        
        for s in all_sets:
            # Check if s can be added to family
            conflict = False
            # Optimization: iterate through pairs in family
            # We can break early if conflict is found
            for i in range(len(family)):
                a = family[i]
                for j in range(i + 1, len(family)):
                    b = family[j]
                    # Check if a, b, s form a sunflower
                    # Core C = a & b
                    # Need a & s == C and b & s == C
                    c = a & b
                    if (a & s == c) and (b & s == c):
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                family.append(s)
                if len(family) >= max_size:
                    break
        
        if len(family) > len(best_family):
            best_family = family
            if len(best_family) >= max_size:
                break

    # Convert bitmasks back to sorted lists of integers
    result = []
    for mask in best_family:
        s = []
        for i in range(1, 17):
            if mask & (1 << (i - 1)):
                s.append(i)
        result.append(s)
        
    # Ensure we don't exceed max_size in return value just in case
    if len(result) > max_size:
        result = result[:max_size]

    return result
