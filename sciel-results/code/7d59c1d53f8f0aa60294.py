import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 57.0  # Leave 3s buffer before 60s kill
    
    # Precompute conflict masks for each x
    # conflict_masks[x] is a list of integers. Each integer represents a set of 3 elements
    # that, if all present in S, would form a 4-AP with x.
    conflict_masks = [None] * (n + 1)
    for x in range(1, n + 1):
        masks = []
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            y1 = x + d
            y2 = x + 2 * d
            y3 = x + 3 * d
            masks.append((1 << y1) | (1 << y2) | (1 << y3))
        
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            y1 = x - d
            y2 = x + d
            y3 = x + 2 * d
            masks.append((1 << y1) | (1 << y2) | (1 << y3))
        
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            y1 = x - 2 * d
            y2 = x - d
            y3 = x + d
            masks.append((1 << y1) | (1 << y2) | (1 << y3))
        
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            y1 = x - 3 * d
            y2 = x - 2 * d
            y3 = x - d
            masks.append((1 << y1) | (1 << y2) | (1 << y3))
        
        # Remove duplicate masks if any
        conflict_masks[x] = list(set(masks))

    # Helper: Check if adding x to mask_S creates a 4-AP
    def can_add(mask_S, x):
        for m in conflict_masks[x]:
            if (m & mask_S) == m:
                return False
        return True

    # Helper: Greedily fill the set given a starting mask and order
    def greedy_fill(start_mask, order):
        mask_S = start_mask
        for x in order:
            if not (mask_S & (1 << x)): # If x is not already in S
                if can_add(mask_S, x):
                    mask_S |= (1 << x)
        return mask_S

    # Convert mask to size
    def mask_size(mask):
        return bin(mask).count("1")

    # Initial Greedy Construction
    best_mask = 0
    best_size = 0
    
    random.seed(42)
    
    # Standard order
    current_mask = greedy_fill(0, list(range(1, n + 1)))
    if mask_size(current_mask) > best_size:
        best_mask = current_mask
        best_size = mask_size(current_mask)
        
    # Reverse order
    current_mask = greedy_fill(0, list(range(n, 0, -1)))
    if mask_size(current_mask) > best_size:
        best_mask = current_mask
        best_size = mask_size(current_mask)
        
    # Random order - generate many diverse seeds
    # With bitsets, this is much faster. We can try more seeds.
    num_random_tries = 500 
    for i in range(num_random_tries):
        if time.time() - start_time > time_limit * 0.2:  # 20% time for init
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current_mask = greedy_fill(0, order)
        if mask_size(current_mask) > best_size:
            best_mask = current_mask
            best_size = mask_size(current_mask)

    # Local Search
    current_mask = best_mask
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, n + 1))
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        # Get candidates not in current_mask
        candidates = [x for x in all_elements if not (current_mask & (1 << x))]
        
        if not candidates:
            break
        
        random.shuffle(candidates)
        
        # Try to add as many as possible in random order (Greedy Add)
        for x in candidates:
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                added = True
                if mask_size(current_mask) > best_size:
                    best_mask = current_mask
                    best_size = mask_size(current_mask)
        
        if not added:
            # Stuck. Perform a kick.
            # Remove 1-5 random elements (increased kick size for more exploration)
            current_size = mask_size(current_mask)
            if current_size > 5:
                k = random.randint(1, 5)
                to_remove = random.sample(all_elements, k)
                for x in to_remove:
                    if current_mask & (1 << x):
                        current_mask &= ~(1 << x)
                
                # Now try to add as many as possible (Greedy Fill)
                for x in all_elements:
                    if not (current_mask & (1 << x)):
                        if can_add(current_mask, x):
                            current_mask |= (1 << x)
                
                if mask_size(current_mask) > best_size:
                    best_mask = current_mask
                    best_size = mask_size(current_mask)

    # Convert best_mask to sorted list of integers
    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
