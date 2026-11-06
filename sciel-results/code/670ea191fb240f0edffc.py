import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55  # safety margin
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    def is_valid_addition(C, family_set):
        """Check if C can be added to family (list of masks)."""
        for i in range(len(family_set)):
            A = family_set[i]
            for j in range(i+1, len(family_set)):
                B = family_set[j]
                I = A & B
                if (A & C) == I and (B & C) == I:
                    return False
        return True
    
    # Greedy initialization with random ordering
    best_family = []
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        random.shuffle(all_masks)
        family = []
        for mask in all_masks:
            if time.time() > deadline - 10:
                break
            if is_valid_addition(mask, family):
                family.append(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Local search / simulated annealing
    family = best_family[:]
    current_size = len(family)
    
    # Precompute for efficiency
    def try_add(fam, C):
        for i in range(len(fam)):
            A = fam[i]
            for j in range(i+1, len(fam)):
                B = fam[j]
                I = A & B
                if (A & C) == I and (B & C) == I:
                    return False
        return True
    
    # Try to add more sets
    added = True
    while added and time.time() < deadline - 5:
        added = False
        for mask in all_masks:
            if mask not in family:
                if try_add(family, mask):
                    family.append(mask)
                    added = True
                    current_size = len(family)
    
    # Swap-based local search
    if time.time() < deadline - 5:
        family_set = set(family)
        T = 1.0
        cooling = 0.999
        
        while time.time() < deadline - 2 and T > 0.01:
            # Try a swap: remove one, add one
            if family:
                idx_out = random.randint(0, len(family)-1)
                mask_out = family[idx_out]
                
                # Find a random mask not in family
                mask_in = random.choice(all_masks)
                attempts = 0
                while mask_in in family_set and attempts < 100:
                    mask_in = random.choice(all_masks)
                    attempts += 1
                if mask_in in family_set:
                    continue
                
                # Check if mask_in can be added to family without mask_out
                fam_temp = family[:idx_out] + family[idx_out+1:]
                if try_add(fam_temp, mask_in):
                    new_family = fam_temp + [mask_in]
                    family = new_family
                    family_set.remove(mask_out)
                    family_set.add(mask_in)
                    current_size = len(family)
                
                # Try pure addition
                if mask_in not in family_set:
                    if try_add(family, mask_in):
                        family.append(mask_in)
                        family_set.add(mask_in)
                        current_size = len(family)
            
            # Try pure removal of a "useless" set? No, we want to maximize size.
            
            T *= cooling
        
        # Final attempt to add
        added = True
        while added and time.time() < deadline - 1:
            added = False
            for mask in all_masks:
                if mask not in family_set:
                    if try_add(family, mask):
                        family.append(mask)
                        family_set.add(mask)
                        added = True
                        current_size = len(family)
    
    # Convert back to sorted lists
    result = []
    for mask in family:
        elements = [i+1 for i in range(16) if mask & (1 << i)]
        result.append(elements)
    
    return result
