import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
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
    
    # Precompute intersection table using numpy for faster access
    masks_arr = np.array(all_masks, dtype=np.int32)
    inter = np.zeros((total, total), dtype=np.int32)
    for i in range(total):
        mi = masks_arr[i]
        for j in range(i + 1, total):
            ij = mi & masks_arr[j]
            inter[i, j] = ij
            inter[j, i] = ij
    
    # Mapping from mask value to index
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}
    
    def check_compatible(family_list, C, C_idx):
        """
        Check if C (at index C_idx) can be added to family_list without forming a sunflower.
        Returns True if compatible.
        """
        if len(family_list) < 2:
            return True
        
        # Bucket family members by their intersection with C
        buckets = {}
        for A in family_list:
            I = int(inter[C_idx, mask_to_idx[A]])
            if I not in buckets:
                buckets[I] = []
            buckets[I].append(A)
        
        # For each bucket, check pairs
        for I, bl in buckets.items():
            if len(bl) < 2:
                continue
            for ia in range(len(bl)):
                A = bl[ia]
                Ai = mask_to_idx[A]
                for ib in range(ia + 1, len(bl)):
                    B = bl[ib]
                    Bi = mask_to_idx[B]
                    if int(inter[Ai, Bi]) == I:
                        return False
        return True

    def greedy_solve(order):
        family_list = []
        family_set = set()
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            C = all_masks[idx]
            if C in family_set:
                continue
            
            if check_compatible(family_list, C, idx):
                family_list.append(C)
                family_set.add(C)
        return family_list

    best_family = []
    best_size = 0
    
    # 1. Lexicographic Order
    order_lex = list(range(total))
    fam = greedy_solve(order_lex)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam
    
    # 2. Reversed Lexicographic Order
    order_rev = list(range(total))
    order_rev.reverse()
    fam = greedy_solve(order_rev)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam
    
    # 3. Multiple random orders
    max_restarts = 300
    rng = random.Random(42)
    
    for i in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        order = list(range(total))
        rng.shuffle(order)
        
        fam = greedy_solve(order)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            if best_size >= 250:
                break
    
    # 4. Local Search / Kick-starts - improved with variable removal
    if time.time() < deadline - 10.0 and best_size > 0:
        for ks_round in range(300):
            if time.time() > deadline - 3.0:
                break
            
            if len(best_family) < 2:
                break
                
            # Remove 1-3 random elements for more diversity
            remove_count = rng.randint(1, min(3, len(best_family) - 1))
            remove_indices = rng.sample(range(len(best_family)), remove_count)
            remove_set = set(remove_indices)
            
            new_family = [x for i, x in enumerate(best_family) if i not in remove_set]
            new_family_set = set(new_family)
            
            # Build candidate list efficiently
            candidates = [m for m in all_masks if m not in new_family_set]
            rng.shuffle(candidates)
            
            improved_family = new_family[:]
            improved_set = new_family_set.copy()
            
            for C in candidates:
                if time.time() > deadline - 3.0:
                    break
                if C in improved_set:
                    continue
                
                C_idx = mask_to_idx[C]
                if check_compatible(improved_family, C, C_idx):
                    improved_family.append(C)
                    improved_set.add(C)
            
            if len(improved_family) > best_size:
                best_size = len(improved_family)
                best_family = improved_family
                if best_size >= 250:
                    break
    
    result = []
    for mask in best_family:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
    
    return result
