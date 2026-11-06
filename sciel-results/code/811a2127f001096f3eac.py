import time
import random

N = 256

def construct():
    # Precomputation
    # For each x in 1..N, store the masks of the other 3 elements for every 4-AP containing x.
    # A 4-AP is (a, b, c, d) with a < b < c < d and b-a = c-b = d-c.
    # If x is a, others are b, c, d. Mask is (1<<b) | (1<<c) | (1<<d).
    # If x is b, others are a, c, d.
    # If x is c, others are a, b, d.
    # If x is d, others are a, b, c.
    
    four_ap_masks = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            
            # Elements are a, b, c, d_val
            # Convert to 0-indexed bits: bit i corresponds to element i+1
            # So element v uses bit (v-1)
            
            # For a (index a-1): others are b, c, d_val
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[a].append(mask_a)
            
            # For b (index b-1): others are a, c, d_val
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[b].append(mask_b)
            
            # For c (index c-1): others are a, b, d_val
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (d_val - 1))
            four_ap_masks[c].append(mask_c)
            
            # For d_val (index d_val-1): others are a, b, c
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            four_ap_masks[d_val].append(mask_d)

    def can_add(A_mask, x):
        # x is 1-indexed
        idx = x - 1
        if (A_mask >> idx) & 1:
            return False # Already in set
            
        for mask in four_ap_masks[x]:
            if (A_mask & mask) == mask:
                return False
        return True

    def count_3aps(A_mask):
        count = 0
        # A 3-AP is (a, b, c) with a < b < c and b-a = c-b = d.
        # c = a + 2d.
        for d in range(1, N // 2 + 1):
            for a in range(1, N - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                    count += 1
        return count

    def add_element(A_mask, x, current_3ap_count):
        # Returns new mask and new count
        # Calculate added 3-APs involving x
        added = 0
        # x is first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N: break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                added += 1
        # x is second: (x-d, x, x+d)
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if a < 1 or c > N: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                added += 1
        # x is third: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                added += 1
                
        new_mask = A_mask | (1 << (x - 1))
        return new_mask, current_3ap_count + added

    def remove_element(A_mask, x, current_3ap_count):
        # Returns new mask and new count
        # Calculate removed 3-APs involving x
        removed = 0
        # x is first
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N: break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                removed += 1
        # x is second
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if a < 1 or c > N: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                removed += 1
        # x is third
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                removed += 1
                
        new_mask = A_mask & ~(1 << (x - 1))
        return new_mask, current_3ap_count - removed

    # Initialize with greedy seed
    A_mask = 0
    for x in range(1, N + 1):
        if can_add(A_mask, x):
            A_mask, _ = add_element(A_mask, x, 0) # We don't care about count yet, just mask
            
    # Calculate initial score
    current_score = count_3aps(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    start_time = time.time()
    timeout = 55.0 # Leave 5s buffer
    
    # Local Search Loop
    improved = True
    while improved and (time.time() - start_time) < timeout:
        improved = False
        # Try removing each element
        for x in range(1, N + 1):
            if (time.time() - start_time) > timeout:
                break
            if not ((A_mask >> (x - 1)) & 1):
                continue
                
            # Remove x
            new_mask, new_score = remove_element(A_mask, x, current_score)
            
            # Try to add elements greedily to new_mask
            # We want to find a maximal superset of new_mask that is valid.
            # To make this faster, we can just iterate through all x not in new_mask and add if possible.
            # This is O(N^2) per removal, which is 256^2 * 256 ~ 16M ops. Might be tight in Python.
            # Optimization: Only try to add elements that were blocked by x?
            # Actually, removing x might unblock many elements.
            
            # Let's do a simple greedy fill
            temp_mask = new_mask
            temp_score = new_score
            for y in range(1, N + 1):
                if not ((temp_mask >> (y - 1)) & 1):
                    if can_add(temp_mask, y):
                        temp_mask, temp_score = add_element(temp_mask, y, temp_score)
                        
            if temp_score > current_score:
                A_mask = temp_mask
                current_score = temp_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                # Restart the loop with the new A_mask
                break # Break inner loop to restart with new state
                
    # Return best found
    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
