import random
import time

def construct():
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # Generate all 4-subsets of [1, 16] as 16-bit masks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    # We want to maximize the size.
    # Randomized greedy
    best_family = []
    
    # Try a few random orders
    # Given the cost, maybe just 1 or 2 orders.
    for attempt in range(5):
        if time.time() - start_time > time_limit:
            break
            
        random.shuffle(all_sets)
        family = []
        S_list = []
        Z_list = []
        
        for c in all_sets:
            if len(family) >= 250:
                break
            
            # Check conflict
            conflict = False
            # Iterate constraints
            # Using range and local vars for speed
            for idx in range(len(S_list)):
                s = S_list[idx]
                if (s & c) == s:
                    z = Z_list[idx]
                    if (c & (~z & 0xFFFF)) == 0:
                        conflict = True
                        break
            
            if not conflict:
                family.append(c)
                
                # Add new constraints
                # c is family[-1]
                # Compare with all previous in family
                for i in range(len(family) - 1):
                    a = family[i]
                    b = c
                    s = a & b
                    diff_ab = a & (~b & 0xFFFF)
                    diff_ba = b & (~a & 0xFFFF)
                    x = (~(diff_ab)) & 0xFFFF
                    y = (~(diff_ba)) & 0xFFFF
                    z = x & y
                    S_list.append(s)
                    Z_list.append(z)
                    
        if len(family) > len(best_family):
            best_family = family
            
    # Convert to list of lists
    result = []
    for mask in best_family:
        subset = []
        for i in range(16):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result
