import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total = len(all_masks)
    
    # Greedy with random restarts to find larger families
    best_family = []
    
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
        
        import random
        random.seed(attempt)
        
        # Shuffle the order
        indices = list(range(total))
        random.shuffle(indices)
        
        family = []
        # For each mask, track which intersections it has with family members
        # Key: (mask_a, mask_b) -> intersection, but we need to check if a third set has same intersection with both
        
        # Efficient check: for a new set C, check against all pairs (A,B) in family
        # if A∩B == A∩C == B∩C, reject C
        
        for idx in indices:
            if time.time() > deadline - 2:
                break
            
            c_mask = all_masks[idx]
            
            # Check against all pairs in family
            compatible = True
            for i in range(len(family)):
                a_mask = family[i]
                a_inter = a_mask & c_mask
                for j in range(i+1, len(family)):
                    b_mask = family[j]
                    b_inter = b_mask & c_mask
                    ab_inter = a_mask & b_mask
                    if a_inter == b_inter == ab_inter:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                family.append(c_mask)
        
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Convert bitmasks back to sorted lists
    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
    
    return result
