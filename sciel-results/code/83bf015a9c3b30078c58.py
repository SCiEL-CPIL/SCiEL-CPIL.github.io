import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # For each element x, store masks of the OTHER elements in APs involving x
    # three_ap_others[x] = list of masks (1<<y | 1<<z) for 3-APs {x, y, z}
    # four_ap_others[x] = list of masks (1<<y | 1<<z | 1<<w) for 4-APs {x, y, z, w}
    
    three_ap_others = [[] for _ in range(N + 1)]
    four_ap_others = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        # 3-APs: a, a+d, a+2d
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            # AP: a, b, c
            mask_bc = (1 << b) | (1 << c)
            mask_ac = (1 << a) | (1 << c)
            mask_ab = (1 << a) | (1 << b)
            
            three_ap_others[a].append(mask_bc)
            three_ap_others[b].append(mask_ac)
            three_ap_others[c].append(mask_ab)
            
            # 4-APs: a, a+d, a+2d, a+3d
            if a + 3 * d <= N:
                e = a + 3 * d
                mask_bce = (1 << b) | (1 << c) | (1 << e)
                mask_ace = (1 << a) | (1 << c) | (1 << e)
                mask_abe = (1 << a) | (1 << b) | (1 << e)
                mask_abc = (1 << a) | (1 << b) | (1 << c)
                
                four_ap_others[a].append(mask_bce)
                four_ap_others[b].append(mask_ace)
                four_ap_others[c].append(mask_abe)
                four_ap_others[e].append(mask_abc)
                
    # Helper: Count 3-APs in a mask
    def count_3aps(A_mask):
        count = 0
        # We need to count each 3-AP once.
        # Iterating over all a, d is O(N^2).
        # Or we can iterate over all precomputed 3-AP masks if we stored them.
        # Let's store all 3-AP masks in a list for counting.
        # But for incremental updates, we use the others lists.
        # For initial count, we can just sum up gain for each element? No, that double counts.
        # A 3-AP {a,b,c} is counted in gain(a), gain(b), gain(c) if we sum gains.
        # So total 3-APs = sum(gain(x)) / 3.
        total_gain = 0
        for x in range(1, N + 1):
            if A_mask & (1 << x):
                for other_mask in three_ap_others[x]:
                    if (A_mask & other_mask) == other_mask:
                        total_gain += 1
        return total_gain // 3

    # Helper: Check if adding x is valid (no 4-AP created)
    def is_valid_add(A_mask, x):
        if A_mask & (1 << x):
            return False
        for other_mask in four_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added to A
    def gain_3aps(A_mask, x):
        if A_mask & (1 << x):
            return 0
        gain = 0
        for other_mask in three_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed from A
    def loss_3aps(A_mask, x):
        if not (A_mask & (1 << x)):
            return 0
        loss = 0
        A_mask_without_x = A_mask & ~(1 << x)
        for other_mask in three_ap_others[x]:
            if (A_mask_without_x & other_mask) == other_mask:
                loss += 1
        return loss

    # Greedy Initialization
    def greedy_init():
        A_mask = 0
        # Sort elements by degree in 3-APs descending
        # Or just random order with validity check
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        for x in elements:
            if is_valid_add(A_mask, x):
                # Add if it creates at least one 3-AP or with some probability
                # To encourage density, we might want to add even if gain is 0 initially
                # but usually high degree elements come first in sorted order.
                # Let's use a sorted order by potential 3-APs
                pass
        
        # Better greedy: Sort by number of 3-APs they can potentially form
        # deg3[x] = len(three_ap_others[x])
        deg3 = [0] * (N + 1)
        for x in range(1, N + 1):
            deg3[x] = len(three_ap_others[x])
        
        elements_sorted = sorted(range(1, N + 1), key=lambda x: deg3[x], reverse=True)
        
        A_mask = 0
        for x in elements_sorted:
            if is_valid_add(A_mask, x):
                A_mask |= (1 << x)
                
        return A_mask

    best_overall_mask = 0
    best_overall_score = 0
    
    def local_search(initial_mask):
        current_mask = initial_mask
        current_score = count_3aps(current_mask)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try Swaps
            # Get list of elements in and out
            in_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            out_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            
            best_swap_delta = 0
            best_swap_in = -1
            best_swap_out = -1
            
            for u in in_elements:
                # Pre-compute loss for u? 
                # Loss is fixed for current mask
                loss_u = loss_3aps(current_mask, u)
                
                for v in out_elements:
                    # Check validity of swap: 
                    # A_temp = current_mask without u
                    # Check if v can be added to A_temp
                    A_temp = current_mask & ~(1 << u)
                    
                    # Check 4-AP constraint for v in A_temp
                    valid = True
                    for other_mask in four_ap_others[v]:
                        if (A_temp & other_mask) == other_mask:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Calculate gain for v in A_temp
                    gain_v = 0
                    for other_mask in three_ap_others[v]:
                        if (A_temp & other_mask) == other_mask:
                            gain_v += 1
                            
                    delta = gain_v - loss_u
                    if delta > best_swap_delta:
                        best_swap_delta = delta
                        best_swap_in = v
                        best_swap_out = u
            
            if best_swap_delta > 0:
                current_mask = (current_mask & ~(1 << best_swap_out)) | (1 << best_swap_in)
                current_score += best_swap_delta
                improved = True
                continue
            
            # 2. Try Adds
            best_add_delta = 0
            best_add_elem = -1
            
            for v in out_elements:
                if is_valid_add(current_mask, v):
                    gain_v = gain_3aps(current_mask, v)
                    if gain_v > best_add_delta:
                        best_add_delta = gain_v
                        best_add_elem = v
                        
            if best_add_delta > 0:
                current_mask |= (1 << best_add_elem)
                current_score += best_add_delta
                improved = True
                continue
            
            # 3. Try Removes (to escape local optima? usually not helpful for max 3-APs unless it allows better swaps)
            # But if we are stuck, maybe removing a low-value element helps?
            # For now, skip pure removals as they decrease score.
            
        return current_mask, current_score

    # Main Loop
    while time.time() < deadline - 5:
        # Generate initial mask
        # Variation: Sometimes random, sometimes greedy
        if random.random() < 0.5:
            init_mask = greedy_init()
        else:
            # Random valid set
            init_mask = 0
            elements = list(range(1, N + 1))
            random.shuffle(elements)
            for x in elements:
                if is_valid_add(init_mask, x):
                    if random.random() < 0.7: # Add with probability
                        init_mask |= (1 << x)
            
        # Local Search
        mask, score = local_search(init_mask)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_mask = mask
            
    # Convert best mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    return result
