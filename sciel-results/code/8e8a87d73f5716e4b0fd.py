import time
import random
import math

N = 256

def construct():
    # Precomputation for 4-APs
    # four_ap_masks[x] contains masks of other 3 elements for each 4-AP containing x
    four_ap_masks = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            
            # 4-AP: a, b, c, d_val
            # Masks for each element representing the other 3
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (d_val - 1))
            four_ap_masks[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (d_val - 1))
            four_ap_masks[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            four_ap_masks[d_val].append(mask_d)

    def count_3aps_involving(x, A_mask):
        count = 0
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N: break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is third: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1: break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                count += 1
        return count

    def can_add(A_mask, x):
        idx = x - 1
        if (A_mask >> idx) & 1:
            return False
        for mask in four_ap_masks[x]:
            if (A_mask & mask) == mask:
                return False
        return True

    def total_3aps(A_mask):
        total = 0
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                total += count_3aps_involving(x, A_mask)
        return total // 3

    # Improved Initialization: Start with full set, remove 4-APs
    A_mask = (1 << N) - 1  # All elements 1..N are set (bits 0..N-1)
    
    # Repair phase: Remove elements involved in 4-APs
    # We iterate and find any 4-AP, remove one element.
    # To be efficient, we can just loop until no 4-AP is found.
    # Since N=256, this should be fast.
    changed = True
    while changed:
        changed = False
        # Find a 4-AP
        found = False
        for d in range(1, N // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                b = a + d
                c = a + 2 * d
                d_val = a + 3 * d
                # Check if all 4 are in A_mask
                if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1) and \
                   ((A_mask >> (c - 1)) & 1) and ((A_mask >> (d_val - 1)) & 1):
                    # Remove one, e.g., the one with highest index or random
                    # Removing 'a' is fine
                    A_mask &= ~(1 << (a - 1))
                    changed = True
                    found = True
                    break
            if found:
                break
    
    current_score = total_3aps(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    start_time = time.time()
    timeout = 55.0
    
    # Simulated Annealing
    T = 10.0  # Start hotter
    cooling = 0.9995
    min_T = 0.01
    
    # Precompute list of all indices for random selection
    all_indices = list(range(N))
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Pick random x_remove and x_add
        # Retry until valid distinct indices
        while True:
            x_remove = random.choice(all_indices) + 1
            if not (A_mask >> (x_remove - 1)) & 1:
                continue
            break
        
        while True:
            x_add = random.choice(all_indices) + 1
            if (A_mask >> (x_add - 1)) & 1:
                continue
            break
        
        # Perform move
        temp_mask = A_mask & ~(1 << (x_remove - 1))
        
        if not can_add(temp_mask, x_add):
            T *= cooling
            continue
            
        new_mask = temp_mask | (1 << (x_add - 1))
        
        removed_count = count_3aps_involving(x_remove, A_mask)
        added_count = count_3aps_involving(x_add, temp_mask)
        
        new_score = current_score - removed_count + added_count
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
        
        T *= cooling

    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
