import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
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
    
    # Build 'others' masks for 3-APs and 4-APs
    three_ap_others = [[] for _ in range(N + 1)]
    for mask3, a, b, c in all_3aps_data:
        three_ap_others[a].append((1 << b) | (1 << c))
        three_ap_others[b].append((1 << a) | (1 << c))
        three_ap_others[c].append((1 << a) | (1 << b))
        
    four_ap_others = [[] for _ in range(N + 1)]
    for mask4, a, b, c, e in all_4aps_data:
        four_ap_others[a].append((1 << b) | (1 << c) | (1 << e))
        four_ap_others[b].append((1 << a) | (1 << c) | (1 << e))
        four_ap_others[c].append((1 << a) | (1 << b) | (1 << e))
        four_ap_others[e].append((1 << a) | (1 << b) | (1 << c))

    # Precompute potential gain for each element relative to a set
    # This is not static, so we compute it on the fly but efficiently.
    
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

    # Greedy Construction with multiple random orderings
    def greedy_construct(seed):
        rng = random.Random(seed)
        A_mask = 0
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
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

    # Local Search: Steepest Ascent with Additions and Swaps
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
            best_swap_remove = -1
            best_swap_add = -1
            best_swap_net = -1
            
            current_elements = [i for i in range(1, N + 1) if A_mask & (1 << i)]
            
            for x_rem in current_elements:
                if time.time() > deadline:
                    break
                loss = loss_3aps(A_mask, x_rem)
                
                for x_add in range(1, N + 1):
                    if A_mask & (1 << x_add):
                        continue
                    
                    A_temp = A_mask & ~(1 << x_rem)
                    
                    valid = True
                    for other_mask in four_ap_others[x_add]:
                        if (A_temp & other_mask) == other_mask:
                            valid = False
                            break
                            
                    if not valid:
                        continue
                        
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
    
    # Run multiple greedy starts with different seeds
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        g_mask = greedy_construct(i * 12345 + 67890)
        l_mask, l_score = local_search(g_mask)
        if l_score > best_overall_score:
            best_overall_score = l_score
            best_overall_mask = l_mask
            
    # If we have time, do iterated local search on the best found
    while time.time() < deadline - 2:
        # Perturb best
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
