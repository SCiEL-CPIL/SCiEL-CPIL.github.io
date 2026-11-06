import time
import random
from itertools import combinations

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 1..16, map to bits 0..15
    all_subsets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << (x - 1))
        all_subsets.append(mask)
    
    n_subsets = len(all_subsets)
    best_family = []
    best_size = 0
    
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for return and overhead
    
    while time.time() < deadline:
        # Randomize order
        indices = list(range(n_subsets))
        random.shuffle(indices)
        
        current_family = []
        
        for idx in indices:
            s = all_subsets[idx]
            is_valid = True
            
            # Check against all pairs in current_family
            # Optimization: if current_family is large, this is slow.
            # But we break early on conflict.
            for i in range(len(current_family)):
                a = current_family[i]
                for j in range(i + 1, len(current_family)):
                    b = current_family[j]
                    c = a & b
                    # Check if a, b, s form a sunflower
                    # Condition: a & b == a & s == b & s
                    if c == (a & s) and c == (b & s):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                current_family.append(s)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family
            
            # Optional: If we reach a very high size, we might stop early?
            # But let's keep searching for a bit longer to be sure.
            # Upper bound is 280. If we get close, we can stop.
            if best_size > 200:
                break

    # Convert bitmasks back to sorted lists of elements
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
