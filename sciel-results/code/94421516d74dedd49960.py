# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: if d is forbidden, -d is also effectively forbidden
    # Actually, the condition is: (a-b)%m not in F and (b-a)%m not in F.
    # Let d = (a-b)%m. Then (b-a)%m = (-d)%m.
    # So d is forbidden if d in F OR (-d)%m in F.
    # This is equivalent to d being in F_sym where F_sym = F union {-f for f in F}.
    
    # Precompute conflict masks
    # conflict_mask[x] is a bitmask where bit y is 1 if x and y conflict.
    conflict_mask = [0] * m
    for x in range(m):
        mask = 0
        for y in range(m):
            if x == y:
                continue
            d = (x - y) % m
            if d in forbidden or (-d) % m in forbidden:
                mask |= (1 << y)
        conflict_mask[x] = mask
        
    def is_valid_addition(mask, x):
        return (mask & conflict_mask[x]) == 0
        
    def get_valid_candidates(mask):
        """Return list of x such that is_valid_addition(mask, x) is True"""
        valid = []
        for x in range(m):
            if not (mask & (1 << x)) and is_valid_addition(mask, x):
                valid.append(x)
        return valid

    def local_search(current_mask, deadline):
        """
        Improves current_mask using add, 1-swap, and 2-swap moves.
        Returns the improved mask.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_size = bin(current_mask).count('1')
            
            # Phase 1: Try to add any possible element
            # We can speed this up by checking all candidates
            for x in range(m):
                if not (current_mask & (1 << x)):
                    if is_valid_addition(current_mask, x):
                        current_mask |= (1 << x)
                        current_size += 1
                        improved = True
            
            # Phase 2: If no additions, try 1-swap moves
            if not improved:
                # For each u in set, for each x not in set:
                # Check if (set - {u}) U {x} is valid.
                # This is equivalent to:
                # 1. x is not in set
                # 2. x is not in conflict with (set - {u})
                #    i.e., (conflict_mask[x] & (current_mask ^ (1 << u))) == 0
                # 3. u is not in conflict with (set - {u}) U {x} ? 
                #    Wait, if we remove u, we don't need to check u's conflicts with the rest anymore.
                #    We just need to ensure the new set is independent.
                #    Since the original set was independent, removing u keeps it independent.
                #    Adding x requires x to be compatible with all remaining elements.
                #    So condition is: (conflict_mask[x] & (current_mask ^ (1 << u))) == 0
                #    And x != u (which is true since x not in set).
                
                set_elements = [i for i in range(m) if (current_mask & (1 << i))]
                not_in_set = [i for i in range(m) if not (current_mask & (1 << i))]
                
                for u in set_elements:
                    u_bit = 1 << u
                    remaining_mask = current_mask ^ u_bit
                    # Find x in not_in_set such that conflict_mask[x] & remaining_mask == 0
                    for x in not_in_set:
                        if is_valid_addition(remaining_mask, x):
                            # Perform swap
                            current_mask = remaining_mask | (1 << x)
                            current_size = bin(current_mask).count('1')
                            improved = True
                            break
                    if improved:
                        break
            
            # Phase 3: If still no improvement, try 2-swap moves
            if not improved:
                set_elements = [i for i in range(m) if (current_mask & (1 << i))]
                not_in_set = [i for i in range(m) if not (current_mask & (1 << i))]
                
                # To keep it fast, we might limit the number of 2-swaps tried or use random sampling
                # But let's try systematic first if size is small enough
                # N=205, set size ~40. Pairs in set ~ 780. Pairs not in set ~ 30k.
                # 780 * 30k is too large for a full scan in Python in a tight loop.
                # We will sample a few 2-swaps.
                
                # Random sampling of 2-swaps
                for _ in range(100): # Try 100 random 2-swaps
                    if len(set_elements) < 2 or len(not_in_set) < 2:
                        break
                    u1, u2 = random.sample(set_elements, 2)
                    x1, x2 = random.sample(not_in_set, 2)
                    
                    mask_remove = current_mask ^ (1 << u1) ^ (1 << u2)
                    
                    # Check if x1, x2 can be added to mask_remove
                    if is_valid_addition(mask_remove, x1) and is_valid_addition(mask_remove | (1 << x1), x2):
                         current_mask = mask_remove | (1 << x1) | (1 << x2)
                         current_size = bin(current_mask).count('1')
                         improved = True
                         break
                         
        return current_mask

    def greedy_construct(order):
        mask = 0
        for x in order:
            if is_valid_addition(mask, x):
                mask |= (1 << x)
        return mask

    best_mask = 0
    best_size = 0
    
    # Deadline guard
    deadline = time.time() + 590.0
    
    # Initial random starts
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(m))
        random.shuffle(order)
        mask = greedy_construct(order)
        mask = local_search(mask, deadline)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Iterated Local Search (ILS)
    # Perturb best solution and re-optimize
    while time.time() < deadline:
        # Perturb best_mask by removing k random elements
        k = random.randint(2, 5)
        set_elements = [i for i in range(m) if (best_mask & (1 << i))]
        
        if len(set_elements) < k:
            # If set is small, just remove all and restart? Or just random restart
            order = list(range(m))
            random.shuffle(order)
            mask = greedy_construct(order)
        else:
            to_remove = random.sample(set_elements, k)
            mask = best_mask
            for u in to_remove:
                mask ^= (1 << u)
                
        mask = local_search(mask, deadline)
        size = bin(mask).count('1')
        
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Convert mask to array
    result = []
    for i in range(m):
        if best_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [0]
        
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
