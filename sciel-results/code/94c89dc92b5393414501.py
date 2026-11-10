# EVOLVE-BLOCK-START
import numpy as np
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # Build forbidden set
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build conflict masks
    # conflict_mask[i] is an integer where bit j is 1 if i and j conflict
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    # Helper: Count bits in an integer
    def popcount(x):
        return bin(x).count('1')
        
    # Helper: Get list of indices from mask
    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]
        
    def fill(mask):
        """
        Greedily add vertices to mask until no more can be added.
        Returns the new mask.
        """
        changed = True
        while changed:
            changed = False
            # Find all candidates v not in mask that can be added
            # v can be added if (conflict_mask[v] & mask) == 0
            for v in range(m):
                if (mask >> v) & 1:
                    continue
                if (conflict_mask[v] & mask) == 0:
                    mask |= (1 << v)
                    changed = True
        return mask

    def try_1_swap(mask):
        """
        Try all 1-swaps (remove one, add one) to see if we can reach a larger maximal set.
        Returns the best mask found.
        """
        best_mask = mask
        best_size = popcount(mask)
        
        # List of elements in mask
        in_mask = mask_to_list(mask)
        
        for u in in_mask:
            # Remove u
            mask_without_u = mask & ~(1 << u)
            
            # Try adding any v not in mask
            for v in range(m):
                if (mask >> v) & 1:
                    continue
                # Check if v can be added to mask_without_u
                # v must not conflict with any node in mask_without_u
                if (conflict_mask[v] & mask_without_u) == 0:
                    new_mask = mask_without_u | (1 << v)
                    # Try to fill this new mask
                    filled_mask = fill(new_mask)
                    size = popcount(filled_mask)
                    if size > best_size:
                        best_size = size
                        best_mask = filled_mask
                        # Early exit? No, we want the best, but if we find m, we can stop.
                        if best_size == m:
                            return best_mask
        return best_mask

    def try_2_swap(mask):
        """
        Try all 2-swaps (remove two, add two) to see if we can reach a larger maximal set.
        This is expensive, so we limit it or use heuristics.
        For N=205, pairs in mask ~ 200^2/2 = 20000.
        Pairs to add ~ 200^2/2 = 20000.
        Total 400M checks. Too slow in Python.
        We will use a randomized 2-swap or limited 2-swap.
        """
        best_mask = mask
        best_size = popcount(mask)
        
        in_mask = mask_to_list(mask)
        not_in_mask = [i for i in range(m) if not ((mask >> i) & 1)]
        
        # Limit the number of swaps to try to keep it fast
        # Or just try a subset
        # Let's try to remove a random pair and add a random pair if it's valid
        # Actually, let's just do a few random 2-swaps
        
        for _ in range(100): # Limit iterations
             if len(in_mask) < 2 or len(not_in_mask) < 2:
                 break
             u1, u2 = random.sample(in_mask, 2)
             v1, v2 = random.sample(not_in_mask, 2)
             
             mask_without = mask & ~(1 << u1) & ~(1 << u2)
             # Check if v1, v2 can be added to mask_without
             # v1 must not conflict with mask_without
             # v2 must not conflict with mask_without
             # v1 must not conflict with v2
             
             if (conflict_mask[v1] & mask_without) == 0 and \
                (conflict_mask[v2] & mask_without) == 0 and \
                (conflict_mask[v1] & (1 << v2)) == 0:
                 new_mask = mask_without | (1 << v1) | (1 << v2)
                 filled_mask = fill(new_mask)
                 size = popcount(filled_mask)
                 if size > best_size:
                     best_size = size
                     best_mask = filled_mask
                     if best_size == m:
                         return best_mask
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Multiple restarts
    for trial in range(200):
        random.seed(trial * 12345 + 67890)
        
        # Random initialization
        # Greedy with random order
        order = list(range(m))
        random.shuffle(order)
        
        current_mask = 0
        for v in order:
            if (conflict_mask[v] & current_mask) == 0:
                current_mask |= (1 << v)
                
        # Fill
        current_mask = fill(current_mask)
        
        # Local Search: 1-swap
        improved = True
        while improved:
            improved = False
            new_mask = try_1_swap(current_mask)
            if popcount(new_mask) > popcount(current_mask):
                current_mask = new_mask
                improved = True
            
            # Try 2-swaps (randomized)
            new_mask = try_2_swap(current_mask)
            if popcount(new_mask) > popcount(current_mask):
                current_mask = new_mask
                improved = True
                
        if popcount(current_mask) > best_size:
            best_size = popcount(current_mask)
            best_mask = current_mask
            
        if best_size == m:
            break

    # Convert best_mask to array
    result_list = mask_to_list(best_mask)
    return np.array(sorted(result_list), dtype=np.int64)

# EVOLVE-BLOCK-END
