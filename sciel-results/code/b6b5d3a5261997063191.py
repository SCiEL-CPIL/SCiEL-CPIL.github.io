import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 60s limit
    
    # Generate all 4-element subsets as bitmasks (elements 0-15)
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    def is_sunflower(A, B, C):
        """Check if A, B, C form a sunflower: A∩B = A∩C = B∩C"""
        ab = A & B
        ac = A & C
        bc = B & C
        return ab == ac == bc
    
    def can_add(C, family_masks):
        """Check if C can be added without creating a sunflower with any pair in family."""
        n = len(family_masks)
        if n < 2:
            return True
        
        # Precompute intersections of C with all family members
        inter = [0] * n
        for i in range(n):
            inter[i] = family_masks[i] & C
            
        # Check all pairs
        for i in range(n):
            A = family_masks[i]
            ac = inter[i]
            
            for j in range(i + 1, n):
                B = family_masks[j]
                bc = inter[j]
                
                # Quick check: if ac != bc, they can't be equal to ab
                if ac != bc:
                    continue
                
                # Now we know ac == bc. Let's call this val.
                # We need A & B == val
                if (A & B) == ac:
                    return False
        return True
    
    # Greedy initialization with random ordering, multiple trials
    best_family = []
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        
        random.shuffle(all_masks)
        family = []
        for mask in all_masks:
            if time.time() > deadline - 15:
                break
            if can_add(mask, family):
                family.append(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search: try to add more sets
    family = best_family[:]
    family_set = set(family)
    
    # Exhaustive add attempt
    added = True
    while added and time.time() < deadline - 10:
        added = False
        for mask in all_masks:
            if mask not in family_set:
                if can_add(mask, family):
                    family.append(mask)
                    family_set.add(mask)
                    added = True
    
    # 2-Swap Local Search (Remove 2, Add 2)
    # This helps escape local optima where 1-swap is stuck.
    if time.time() < deadline - 5:
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 3 and T > 0.01:
            # Try 2-swap
            if len(family) >= 2:
                # Pick 2 random indices to remove
                idx1, idx2 = random.sample(range(len(family)), 2)
                mask_out1 = family[idx1]
                mask_out2 = family[idx2]
                
                # Find 2 random masks to add
                mask_in1 = random.choice(all_masks)
                mask_in2 = random.choice(all_masks)
                
                # Ensure they are distinct and not in family (except the ones being removed)
                attempts = 0
                while (mask_in1 in family_set or mask_in2 in family_set or mask_in1 == mask_in2) and attempts < 50:
                    mask_in1 = random.choice(all_masks)
                    mask_in2 = random.choice(all_masks)
                    attempts += 1
                    
                if mask_in1 in family_set or mask_in2 in family_set or mask_in1 == mask_in2:
                    continue
                
                # Check if {mask_in1, mask_in2} can be added to family \ {mask_out1, mask_out2}
                F_prime = [m for i, m in enumerate(family) if i != idx1 and i != idx2]
                
                if can_add(mask_in1, F_prime + [mask_in2]) and can_add(mask_in2, F_prime + [mask_in1]):
                    # Accept move
                    new_family = F_prime + [mask_in1, mask_in2]
                    family = new_family
                    family_set.remove(mask_out1)
                    family_set.remove(mask_out2)
                    family_set.add(mask_in1)
                    family_set.add(mask_in2)
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for mask in all_masks:
                if mask not in family_set:
                    if can_add(mask, family):
                        family.append(mask)
                        family_set.add(mask)
                        added = True
    
    # Convert back to sorted lists (1-indexed)
    result = []
    for mask in family:
        elements = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result
