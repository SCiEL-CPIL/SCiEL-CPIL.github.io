import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return a & b == a & c == b & c

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Pre-compute all 4-element subsets of [1,16] as bitmasks for speed
    # and also keep the original tuples for output
    all_sets_info = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i-1))
        all_sets_info.append((mask, row))
    
    n_total = len(all_sets_info)
    all_masks = [info[0] for info in all_sets_info]
    all_rows = [info[1] for info in all_sets_info]
    
    # Helper: check if mask can be added to a set of chosen masks
    def can_add_mask(mask, chosen_masks):
        n = len(chosen_masks)
        for i in range(n):
            ma = chosen_masks[i]
            for j in range(i + 1, n):
                mb = chosen_masks[j]
                if (ma & mb) == (ma & mask) == (mb & mask):
                    return False
        return True
    
    # Helper: run greedy given an order of indices
    def greedy_pass(order):
        chosen_masks = []
        chosen_indices = []
        for idx in order:
            mask = all_masks[idx]
            if can_add_mask(mask, chosen_masks):
                chosen_masks.append(mask)
                chosen_indices.append(idx)
        return chosen_indices
    
    # Local search: try to improve by swapping one chosen set with one unchosen set
    def local_search(chosen_indices, max_iters=500):
        chosen_set = set(chosen_indices)
        chosen_masks = [all_masks[i] for i in chosen_indices]
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > deadline - 5:
                break
            
            # Try each swap: remove one chosen, add one unchosen
            for out_idx in range(len(chosen_indices)):
                if time.time() > deadline - 5:
                    break
                out_mask = chosen_masks[out_idx]
                # Temporarily remove out_idx
                temp_chosen_masks = [chosen_masks[k] for k in range(len(chosen_masks)) if k != out_idx]
                
                for in_idx in unchosen:
                    in_mask = all_masks[in_idx]
                    if can_add_mask(in_mask, temp_chosen_masks):
                        # Perform the swap
                        chosen_indices[out_idx] = in_idx
                        chosen_masks[out_idx] = in_mask
                        chosen_set.discard(chosen_indices[out_idx])
                        chosen_set.add(in_idx)
                        # Remove in_idx from unchosen, add old chosen idx
                        unchosen.remove(in_idx)
                        unchosen.append(out_idx)
                        improved = True
                        break
                
                if improved:
                    break
        
        return chosen_indices
    
    best_family = []
    best_size = 0
    
    # Define several deterministic orderings
    indices = list(range(n_total))
    
    # 1. Lexicographic (default)
    order_1 = indices[:]
    
    # 2. Reverse Lexicographic
    order_2 = indices[::-1]
    
    # 3. Sorted by bitmask value
    order_3 = sorted(indices, key=lambda i: all_masks[i])
    
    # 4. Sorted by sum of elements
    order_4 = sorted(indices, key=lambda i: sum(all_rows[i]))
    
    # 5. Sorted by number of elements in even positions (heuristic)
    order_5 = sorted(indices, key=lambda i: sum(1 for x in all_rows[i] if x % 2 == 0))
    
    # 6. Sorted by product of elements
    order_6 = sorted(indices, key=lambda i: all_rows[i][0] * all_rows[i][3])
    
    orders = [order_1, order_2, order_3, order_4, order_5, order_6]
    
    # Add a couple of random orders just in case, but prioritize deterministic
    for _ in range(2):
        order_r = indices[:]
        random.shuffle(order_r)
        orders.append(order_r)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        family_indices = greedy_pass(order)
        
        # Apply local search to improve
        if time.time() < deadline - 10:
            family_indices = local_search(family_indices)
        
        family = [all_rows[i] for i in family_indices]
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
        # If we hit the cap, we can stop
        if best_size >= 250:
            break
            
    return [sorted(s) for s in best_family]
