import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Generate all 4-element subsets of [1, 16] as bitmasks
    all_masks = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_total = len(all_masks)  # 1820
    
    best_family = []
    best_size = 0
    
    def is_valid_addition(family, new_mask):
        """Check if adding new_mask to family creates a sunflower."""
        # For every pair (a, b) in family, check if a&b == a&new_mask == b&new_mask
        for i in range(len(family)):
            a = family[i]
            a_new = a & new_mask
            for j in range(i + 1, len(family)):
                b = family[j]
                b_new = b & new_mask
                if a_new == b_new and (a & b) == a_new:
                    return False
        return True
    
    # Multiple random greedy passes with different seeds
    rng = random.Random(42)
    
    for iteration in range(500):
        if time.time() > deadline - 10:
            break
        
        shuffled = list(range(n_total))
        rng.shuffle(shuffled)
        
        family = []
        for idx in shuffled:
            m = all_masks[idx]
            if is_valid_addition(family, m):
                family.append(m)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
        
        # If we've reached 250, we can't do better
        if best_size >= 250:
            break
    
    # Local search: try to add any remaining mask that fits
    if time.time() < deadline - 5:
        family_set = set(best_family)
        family = list(best_family)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any mask not in family
            for idx in range(n_total):
                if time.time() > deadline - 2:
                    break
                m = all_masks[idx]
                if m in family_set:
                    continue
                if is_valid_addition(family, m):
                    family.append(m)
                    family_set.add(m)
                    improved = True
                    if len(family) >= 250:
                        break
            
            # If still room, try swaps: remove one, add one
            if len(family) < 250 and time.time() < deadline - 2:
                for i in range(len(family)):
                    if time.time() > deadline - 2:
                        break
                    removed = family[i]
                    family_no_i = family[:i] + family[i+1:]
                    for idx in range(n_total):
                        if time.time() > deadline - 2:
                            break
                        m = all_masks[idx]
                        if m in family_set:
                            continue
                        if is_valid_addition(family_no_i, m):
                            family[i] = m
                            family_set.discard(removed)
                            family_set.add(m)
                            improved = True
                            break
                    if improved:
                        break
    
    if len(family) > best_size:
        best_family = family
    
    # Convert bitmasks back to sorted lists of 1-indexed elements
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result

if __name__ == "__main__":
    # Just for testing, not part of the evaluation call
    res = construct()
    print(len(res))
