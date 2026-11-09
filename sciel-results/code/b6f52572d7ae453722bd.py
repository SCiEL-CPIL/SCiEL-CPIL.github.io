import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    # 1. Precompute conflict masks
    # S is represented as a bitmask of N bits.
    # Element x (1-indexed) corresponds to bit (x-1).
    # conflict_masks[x] is a list of bitmasks. Each mask represents the other 3 elements
    # in a 4-AP that includes x.
    # If (S & mask) == mask, then adding x is forbidden.
    
    # Initialize list of lists for 1..N
    conflict_masks = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d
    # a >= 1, a+3d <= N => a <= N - 3d
    # d >= 1
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d  # Using e instead of d to avoid confusion with step d
            
            # Masks for each element
            # For a: need b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            # For b: need a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            # For c: need a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            # For e: need a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            
            conflict_masks[a].append(mask_a)
            conflict_masks[b].append(mask_b)
            conflict_masks[c].append(mask_c)
            conflict_masks[e].append(mask_e)
            
    # Optimization: Convert lists to tuples for slightly faster iteration if needed,
    # but lists are fine.
    
    def can_add(x, S_mask):
        """Check if x can be added to the set represented by S_mask."""
        for mask in conflict_masks[x]:
            if (S_mask & mask) == mask:
                return False
        return True
        
    def get_size(S_mask):
        return bin(S_mask).count('1')
        
    def greedy_construction(order, S_mask=0):
        """Greedy fill given an order of elements."""
        for x in order:
            if not (S_mask & (1 << (x - 1))):
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
        return S_mask
        
    def local_search(S_mask, max_iters=1000):
        """
        Hill climbing / Local search.
        1. Try to add any missing element.
        2. Try swaps (remove y, add x).
        """
        current_size = get_size(S_mask)
        elements = list(range(1, N + 1))
        
        # Precompute list of present and missing elements for faster access?
        # With bitmask, we can just iterate 1..N. N=512 is small.
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Phase 1: Add any possible element
            for x in elements:
                if not (S_mask & (1 << (x - 1))):
                    if can_add(x, S_mask):
                        S_mask |= (1 << (x - 1))
                        current_size += 1
                        improved = True
            if improved:
                continue
                
            # Phase 2: Try swaps
            # Pick a random present element y and a random missing element x
            # To make it efficient, we can sample.
            present = [x for x in elements if (S_mask & (1 << (x - 1)))]
            missing = [x for x in elements if not (S_mask & (1 << (x - 1)))]
            
            if not missing:
                break
                
            # Try a few random swaps
            for _ in range(50):
                if time.time() > deadline:
                    return S_mask, current_size
                    
                y = random.choice(present)
                x = random.choice(missing)
                
                # Temporarily remove y
                S_mask_new = S_mask & ~(1 << (y - 1))
                
                # Check if x can be added
                if can_add(x, S_mask_new):
                    S_mask_new |= (1 << (x - 1))
                    # Size is the same, but structure changed.
                    # We might be able to add more now.
                    # Try to add any other missing element
                    added_more = False
                    for z in missing:
                        if z != x and not (S_mask_new & (1 << (z - 1))):
                            if can_add(z, S_mask_new):
                                S_mask_new |= (1 << (z - 1))
                                current_size += 1
                                added_more = True
                                
                    if added_more:
                        S_mask = S_mask_new
                        improved = True
                        # Update present/missing lists for next iteration? 
                        # It's safer to recompute at start of loop, but that's O(N).
                        # For now, let's just break inner loop and restart outer.
                        break
                    else:
                        # No immediate gain, but maybe this swap enables future gains?
                        # In standard hill climbing, we only accept if size increases.
                        # If size doesn't increase, we revert.
                        pass
            
        return S_mask, current_size

    def simulated_annealing(S_mask, steps=5000, T_start=1.0, T_end=0.01):
        """
        Simulated Annealing.
        Moves:
        1. Add x (if possible) -> Accept always (improvement)
        2. Remove y -> Accept with prob exp(-1/T) (worsening)
        3. Swap y->x -> Accept with prob exp(0/T) if size same? No, usually we want to maximize size.
           If size decreases, accept with prob exp(-delta/T).
           If size increases, accept always.
        """
        current_size = get_size(S_mask)
        elements = list(range(1, N + 1))
        
        for i in range(steps):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / steps)
            
            # Choose a move
            r = random.random()
            if r < 0.5:
                # Try to add a random missing element
                missing = [x for x in elements if not (S_mask & (1 << (x - 1)))]
                if not missing:
                    continue
                x = random.choice(missing)
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    current_size += 1
                else:
                    # Can't add, maybe try removing something to allow it?
                    # That's a complex move. Skip for now.
                    pass
            elif r < 0.8:
                # Try to remove a random present element
                present = [x for x in elements if (S_mask & (1 << (x - 1)))]
                if not present:
                    continue
                y = random.choice(present)
                # Worsening move
                if random.random() < (T / T_start): # Simple cooling acceptance
                    S_mask &= ~(1 << (y - 1))
                    current_size -= 1
            else:
                # Try swap
                present = [x for x in elements if (S_mask & (1 << (x - 1)))]
                missing = [x for x in elements if not (S_mask & (1 << (x - 1)))]
                if not present or not missing:
                    continue
                y = random.choice(present)
                x = random.choice(missing)
                
                S_mask_new = S_mask & ~(1 << (y - 1))
                if can_add(x, S_mask_new):
                    S_mask_new |= (1 << (x - 1))
                    # Size is same. 
                    # In strict maximization, we don't accept neutral moves unless we think it helps escape.
                    # Let's accept with prob T to allow exploration.
                    if random.random() < (T / T_start):
                        S_mask = S_mask_new
                        # Try to fill up again
                        for z in missing:
                            if z != x and not (S_mask & (1 << (z - 1))):
                                if can_add(z, S_mask):
                                    S_mask |= (1 << (z - 1))
                                    current_size += 1
                        # If size increased, it's a good move.
                        # If size stayed same, it's an exploration move.
                        # If size decreased (shouldn't happen with this logic), revert.
                        if get_size(S_mask) < current_size:
                             # Revert
                             # This logic is getting complicated. 
                             # Let's stick to simple hill climbing for robustness.
                             pass

        return S_mask, current_size

    best_S_mask = 0
    best_size = 0
    
    # Strategy 1: Multiple Greedy Random Restarts
    rng = random.Random(42)
    
    # 1. Natural Order
    S_mask = greedy_construction(range(1, N + 1))
    size = get_size(S_mask)
    if size > best_size:
        best_size = size
        best_S_mask = S_mask
        
    # 2. Reverse Order
    S_mask = greedy_construction(range(N, 0, -1))
    size = get_size(S_mask)
    if size > best_size:
        best_size = size
        best_S_mask = S_mask
        
    # 3. Random Orders
    for i in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_mask = greedy_construction(order)
        size = get_size(S_mask)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
            
        # Local Search on this result
        S_mask, size = local_search(S_mask, max_iters=20)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
            
    # Strategy 2: Intensive Local Search on the Best Found So Far
    if best_S_mask:
        S_mask, size = local_search(best_S_mask, max_iters=500)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
            
    # Strategy 3: More Restarts with Local Search
    for i in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_mask = greedy_construction(order)
        S_mask, size = local_search(S_mask, max_iters=50)
        if size > best_size:
            best_size = size
            best_S_mask = S_mask
            
    # Final Polish: Try to add any remaining elements to the best solution
    for x in range(1, N + 1):
        if not (best_S_mask & (1 << (x - 1))):
            if can_add(x, best_S_mask):
                best_S_mask |= (1 << (x - 1))
                best_size += 1
                
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_S_mask & (1 << (x - 1)))]
    
    return result

# EVOLVE-BLOCK-END
