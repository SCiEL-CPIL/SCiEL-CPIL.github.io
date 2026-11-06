import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs as bitmasks
    # Bit i represents element i+1 (1-indexed)
    
    all_3aps = []
    all_4aps = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            mask3 = (1 << a) | (1 << b) | (1 << c)
            all_3aps.append(mask3)
            
            if a + 3 * d <= N:
                e = a + 3 * d
                mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << e)
                all_4aps.append(mask4)
    
    # Precompute neighbor masks for fast validity and gain checks
    # For each x, list of masks of the *other* elements in 3-APs containing x
    three_ap_others = [[] for _ in range(N + 1)]
    for mask3 in all_3aps:
        # Find which elements are in this mask
        # Since N=256, we can iterate or use bit tricks.
        # Simpler: just check which of 1..N are set.
        # But we generated them, so we can just store the elements too or re-derive.
        # Let's re-derive elements from mask for simplicity in precomp loop, 
        # or better, store elements alongside mask in a list.
        pass
        
    # Let's redo precomp to store elements for mapping to 'others'
    all_3aps_data = [] # (mask, a, b, c)
    all_4aps_data = [] # (mask, a, b, c, e)
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            mask3 = (1 << a) | (1 << b) | (1 << c)
            all_3aps_data.append((mask3, a, b, c))
            
            if a + 3 * d <= N:
                e = a + 3 * d
                mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << e)
                all_4aps_data.append((mask4, a, b, c, e))
    
    # Build 'others' masks
    # three_ap_others[x] contains masks of the other 2 elements for each 3-AP with x
    three_ap_others = [[] for _ in range(N + 1)]
    for mask3, a, b, c in all_3aps_data:
        # For a, others are b, c
        three_ap_others[a].append((1 << b) | (1 << c))
        # For b, others are a, c
        three_ap_others[b].append((1 << a) | (1 << c))
        # For c, others are a, b
        three_ap_others[c].append((1 << a) | (1 << b))
        
    # four_ap_others[x] contains masks of the other 3 elements for each 4-AP with x
    four_ap_others = [[] for _ in range(N + 1)]
    for mask4, a, b, c, e in all_4aps_data:
        # For a, others are b, c, e
        four_ap_others[a].append((1 << b) | (1 << c) | (1 << e))
        # For b, others are a, c, e
        four_ap_others[b].append((1 << a) | (1 << c) | (1 << e))
        # For c, others are a, b, e
        four_ap_others[c].append((1 << a) | (1 << b) | (1 << e))
        # For e, others are a, b, c
        four_ap_others[e].append((1 << a) | (1 << b) | (1 << c))

    # Helper: Check if adding x is valid (no 4-AP created)
    def is_valid_add(A_mask, x):
        if A_mask & (1 << x):
            return False
        for other_mask in four_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added
    def gain_3aps(A_mask, x):
        if A_mask & (1 << x):
            return 0
        gain = 0
        for other_mask in three_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed
    def loss_3aps(A_mask, x):
        if not (A_mask & (1 << x)):
            return 0
        loss = 0
        # When removing x, we lose 3-APs where x is present and the other 2 are in A.
        # Since x is in A, the condition is (A_mask without x) & other_mask == other_mask
        A_mask_without_x = A_mask & ~(1 << x)
        for other_mask in three_ap_others[x]:
            if (A_mask_without_x & other_mask) == other_mask:
                loss += 1
        return loss

    # Count total 3-APs in A_mask
    def count_3aps(A_mask):
        count = 0
        for mask3, a, b, c in all_3aps_data:
            if (A_mask & mask3) == mask3:
                count += 1
        return count

    # Greedy Construction
    def greedy_construct():
        A_mask = 0
        # Simple greedy: add elements in random order if valid and gain > 0
        # Or steepest ascent. Steepest ascent is better for quality.
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        while remaining:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            for x in remaining:
                if not is_valid_add(A_mask, x):
                    continue
                g = gain_3aps(A_mask, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
                
            A_mask |= (1 << best_x)
            remaining.remove(best_x)
            
        return A_mask

    # Local Search
    def local_search(initial_mask):
        A_mask = initial_mask
        current_score = count_3aps(A_mask)
        best_score = current_score
        best_mask = A_mask
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            # 1. Try adding elements
            best_add_x = -1
            best_add_gain = -1
            for x in range(1, N + 1):
                if A_mask & (1 << x):
                    continue
                if not is_valid_add(A_mask, x):
                    continue
                g = gain_3aps(A_mask, x)
                if g > best_add_gain:
                    best_add_gain = g
                    best_add_x = x
                    
            if best_add_x != -1 and best_add_gain > 0:
                A_mask |= (1 << best_add_x)
                current_score += best_add_gain
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                improved = True
                continue # Restart to re-evaluate additions
                
            # 2. Try swaps
            # To make this fast, we only consider removing elements that have low loss
            # and adding elements that have high gain.
            # However, simple steepest ascent swap is O(N^2 * checks).
            # N=256, N^2 ~ 65k. Checks are bitwise. Should be fast enough.
            
            best_swap_remove = -1
            best_swap_add = -1
            best_swap_net = -1
            
            # Get list of elements in A
            current_elements = [i for i in range(1, N + 1) if A_mask & (1 << i)]
            
            for x_rem in current_elements:
                if time.time() > deadline:
                    break
                loss = loss_3aps(A_mask, x_rem)
                
                # If loss is already high, skip? No, net gain matters.
                # But if loss is huge, it's unlikely to be best unless gain is even huger.
                
                # Try adding every x_add not in A
                for x_add in range(1, N + 1):
                    if A_mask & (1 << x_add):
                        continue
                    # Check validity of adding x_add to (A \ {x_rem})
                    # This is tricky. is_valid_add checks against A_mask.
                    # We need to check if adding x_add to A_mask without x_rem creates a 4-AP.
                    # A 4-AP is created if there are 3 other elements in (A \ {x_rem}) that form a 4-AP with x_add.
                    # So, check if any 4-AP containing x_add has its other 3 elements in (A_mask & ~(1<<x_rem)).
                    
                    A_temp = A_mask & ~(1 << x_rem)
                    
                    valid = True
                    for other_mask in four_ap_others[x_add]:
                        if (A_temp & other_mask) == other_mask:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
                    # Calculate gain of adding x_add to A_temp
                    # Gain is number of 3-APs containing x_add where other 2 are in A_temp
                    gain = 0
                    for other_mask in three_ap_others[x_add]:
                        if (A_temp & other_mask) == other_mask:
                            gain += 1
                            
                    net = gain - loss
                    if net > best_swap_net:
                        best_swap_net = net
                        best_swap_remove = x_rem
                        best_swap_add = x_add
                        
            if best_swap_remove != -1 and best_swap_net > 0:
                A_mask &= ~(1 << best_swap_remove)
                A_mask |= (1 << best_swap_add)
                current_score += best_swap_net
                if current_score > best_score:
                    best_score = current_score
                    best_mask = A_mask
                improved = True
                
        return best_mask, best_score

    # Main Logic
    best_overall_mask = 0
    best_overall_score = 0
    
    # Run multiple greedy starts
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave time for local search
            break
        g_mask = greedy_construct()
        l_mask, l_score = local_search(g_mask)
        if l_score > best_overall_score:
            best_overall_score = l_score
            best_overall_mask = l_mask
            
    # If we have time, do iterated local search on the best found
    while time.time() < deadline - 2:
        # Perturb best
        # Remove 1-3 random elements
        k = random.randint(1, 3)
        elements_in_best = [i for i in range(1, N + 1) if best_overall_mask & (1 << i)]
        if len(elements_in_best) > k:
            to_remove = random.sample(elements_in_best, k)
            perturbed_mask = best_overall_mask
            for x in to_remove:
                perturbed_mask &= ~(1 << x)
        else:
            continue
            
        l_mask, l_score = local_search(perturbed_mask)
        if l_score > best_overall_score:
            best_overall_score = l_score
            best_overall_mask = l_mask

    # Convert best mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    return result
