import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    # We store them as (mask, tuple_of_elements) for easy conversion back
    all_masks = []
    all_tuples = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        all_tuples.append(combo)
        
    total = len(all_masks)
    
    # Map mask to index
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    def run_greedy(order_indices):
        family_masks = []
        family_set = set() # For O(1) check if mask is already in family? 
                           # Actually, we iterate unique indices, so no duplicates in input.
                           # But we need to check compatibility.
        
        for idx in order_indices:
            if time.time() > deadline - 2:
                break
            c_mask = all_masks[idx]
            
            # Check against all pairs in family
            compatible = True
            # Optimization: If family is large, this is the bottleneck.
            # We can break early.
            for i in range(len(family_masks)):
                a_mask = family_masks[i]
                # Precompute a & c? No, b changes.
                ac = a_mask & c_mask
                for j in range(i + 1, len(family_masks)):
                    b_mask = family_masks[j]
                    ab = a_mask & b_mask
                    bc = b_mask & c_mask
                    if ab == ac == bc:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                family_masks.append(c_mask)
                
        return family_masks

    def polish(family_masks, remaining_indices, max_iter=100):
        """
        Try to add more sets from remaining_indices to family_masks.
        Returns a new family list.
        """
        fam = family_masks[:]
        # Shuffle remaining indices to break any structure
        rem = remaining_indices[:]
        random.shuffle(rem)
        
        added = 0
        for idx in rem:
            if time.time() > deadline - 2:
                break
            if added > max_iter:
                break # Limit polish work per call? Or just let it run?
            
            c_mask = all_masks[idx]
            
            # Check compatibility
            compatible = True
            for i in range(len(fam)):
                a_mask = fam[i]
                ac = a_mask & c_mask
                for j in range(i + 1, len(fam)):
                    b_mask = fam[j]
                    ab = a_mask & b_mask
                    bc = b_mask & c_mask
                    if ab == ac == bc:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                fam.append(c_mask)
                added += 1
                
        return fam

    best_family = []
    
    # 1. Standard Lexicographic Order
    order = list(range(total))
    fam = run_greedy(order)
    if len(fam) > len(best_family):
        best_family = fam[:]
        
    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutations of the Universe
    for attempt in range(20): # Increased from 10
        if time.time() > deadline - 5:
            break
        
        random.seed(42 + attempt)
        perm = list(range(n))
        random.shuffle(perm)
        
        # Create a sorting key for each mask
        keys = []
        for idx in range(total):
            mask = all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            elems_sorted_by_perm = sorted(elems)
            keys.append((tuple(elems_sorted_by_perm), idx))
            
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 4. Direct Random Shuffling of Candidates
    for attempt in range(30): # Increased from 0
        if time.time() > deadline - 5:
            break
        
        random.seed(1000 + attempt)
        order_rand = list(range(total))
        random.shuffle(order_rand)
        
        fam = run_greedy(order_rand)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 5. Polish the best family found
    if best_family and time.time() < deadline - 5:
        best_set = set(best_family)
        remaining = [i for i in range(total) if all_masks[i] not in best_set]
        
        # Try a few polish passes with different random seeds for the shuffle
        for p_attempt in range(5):
            if time.time() > deadline - 2:
                break
            random.seed(2000 + p_attempt)
            polished = polish(best_family, remaining)
            if len(polished) > len(best_family):
                best_family = polished
                # Update remaining?
                best_set = set(best_family)
                remaining = [i for i in range(total) if all_masks[i] not in best_set]

    # Convert best_family masks back to lists of 1-based integers
    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result
