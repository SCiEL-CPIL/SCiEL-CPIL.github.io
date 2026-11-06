import time
import random
import math

def construct():
    deadline = time.time() + 54  # Safety margin for 90s limit
    N = 256
    
    # Precomputation using 0-indexed internally, 1-indexed for output
    three_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a - 1, a + d - 1, a + 2 * d - 1))
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Build element-to-AP-index mappings
    elem_to_3aps = [[] for _ in range(N)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
    
    elem_to_4aps = [[] for _ in range(N)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # Precompute bitmask for each AP (for fast "all in set" checks)
    # For 3-APs: bitmask of the 3 elements
    three_ap_masks = []
    for a, b, c in three_aps:
        mask = (1 << a) | (1 << b) | (1 << c)
        three_ap_masks.append(mask)
    
    # For 4-APs: bitmask of the 4 elements
    four_ap_masks = []
    for a, b, c, d in four_aps:
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        four_ap_masks.append(mask)
    
    # For each element, precompute bitmask of "other two" in each 3-AP it belongs to
    # This allows fast delta_add calculation
    elem_3ap_other_masks = [[] for _ in range(N)]
    for x in range(N):
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                other_mask = (1 << b) | (1 << c)
            elif x == b:
                other_mask = (1 << a) | (1 << c)
            else:
                other_mask = (1 << a) | (1 << b)
            elem_3ap_other_masks[x].append(other_mask)
    
    # For each element, precompute bitmask of "other three" in each 4-AP it belongs to
    elem_4ap_other_masks = [[] for _ in range(N)]
    for x in range(N):
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                other_mask = (1 << b) | (1 << c) | (1 << d)
            elif x == b:
                other_mask = (1 << a) | (1 << c) | (1 << d)
            elif x == c:
                other_mask = (1 << a) | (1 << b) | (1 << d)
            else:
                other_mask = (1 << a) | (1 << b) | (1 << c)
            elem_4ap_other_masks[x].append(other_mask)
    
    def count_3aps(mask):
        count = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                count += 1
        return count
    
    def delta_add(mask, x):
        delta = 0
        for other_mask in elem_3ap_other_masks[x]:
            if (mask & other_mask) == other_mask:
                delta += 1
        return delta
    
    def delta_remove(mask, x):
        delta = 0
        for other_mask in elem_3ap_other_masks[x]:
            if (mask & other_mask) == other_mask:
                delta += 1
        return delta
    
    def is_valid_add(mask, x):
        for other_mask in elem_4ap_other_masks[x]:
            if (mask & other_mask) == other_mask:
                return False
        return True
    
    def is_valid_swap(mask, x_out, x_in):
        # Check if x_in creates a 4-AP with (mask without x_out)
        # A 4-AP containing x_in is invalid if other 3 are in mask AND none is x_out
        x_out_bit = 1 << x_out
        reduced_mask = mask & ~x_out_bit
        for other_mask in elem_4ap_other_masks[x_in]:
            if (reduced_mask & other_mask) == other_mask:
                return False
        return True
    
    def greedy_init():
        mask = 0
        for x in range(N):
            if is_valid_add(mask, x):
                mask |= (1 << x)
        return mask
    
    def simulated_annealing(init_mask, time_budget):
        random.seed(42)
        T = 5.0
        cooling = 0.998
        min_T = 0.005
        deadline_sa = time.time() + time_budget
        
        mask = init_mask
        best_mask = mask
        best_score = count_3aps(best_mask)
        current_score = best_score
        
        while time.time() < deadline_sa:
            if T < min_T:
                T = min_T
            
            move_type = random.random()
            
            if move_type < 0.35:  # Add
                # Find candidates that can be added
                candidates = []
                for x in range(N):
                    if not (mask >> x) & 1:
                        if is_valid_add(mask, x):
                            candidates.append(x)
                if candidates:
                    x = random.choice(candidates)
                    delta = delta_add(mask, x)
                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.3):
                        mask |= (1 << x)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = mask
                            
            elif move_type < 0.55:  # Remove
                if mask:
                    # Find elements in mask
                    in_mask = [x for x in range(N) if (mask >> x) & 1]
                    if in_mask:
                        x = random.choice(in_mask)
                        delta = -delta_remove(mask, x)
                        if random.random() < math.exp(delta / T):
                            mask &= ~(1 << x)
                            current_score += delta
                            
            else:  # Swap
                if mask:
                    in_mask = [x for x in range(N) if (mask >> x) & 1]
                    not_in_mask = [x for x in range(N) if not (mask >> x) & 1]
                    if in_mask and not_in_mask:
                        x_out = random.choice(in_mask)
                        x_in = random.choice(not_in_mask)
                        if is_valid_swap(mask, x_out, x_in):
                            loss = delta_remove(mask, x_out)
                            # Calculate gain: delta_add for x_in in (mask without x_out)
                            reduced_mask = mask & ~(1 << x_out)
                            gain = delta_add(reduced_mask, x_in)
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                mask &= ~(1 << x_out)
                                mask |= (1 << x_in)
                                current_score += delta
                                if current_score > best_score:
                                    best_score = current_score
                                    best_mask = mask
                                
            T *= cooling
            
        return best_mask, best_score
    
    # Multi-restart strategy
    best_overall_mask = None
    best_overall_score = -1
    
    # First, do a greedy init and SA
    t_start = time.time()
    init_mask = greedy_init()
    time_elapsed = time.time() - t_start
    remaining = deadline - time.time()
    
    # Split time among restarts
    num_restarts = 3
    time_per_restart = max(5, (remaining - 5) / num_restarts)  # 5s safety
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5:
            break
        
        if restart == 0:
            init_mask = greedy_init()
        else:
            # Perturb the best found so far
            if best_overall_mask is None:
                init_mask = greedy_init()
            else:
                # Randomly remove some elements and re-add
                mask = best_overall_mask
                in_mask = [x for x in range(N) if (mask >> x) & 1]
                # Remove 5-10 random elements
                num_remove = random.randint(5, 10)
                to_remove = random.sample(in_mask, min(num_remove, len(in_mask)))
                for x in to_remove:
                    mask &= ~(1 << x)
                # Re-add greedily
                for x in range(N):
                    if not (mask >> x) & 1:
                        if is_valid_add(mask, x):
                            mask |= (1 << x)
                init_mask = mask
        
        t0 = time.time()
        result_mask, result_score = simulated_annealing(init_mask, time_per_restart)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall_mask = result_mask
    
    # Convert back to 1-indexed sorted list
    if best_overall_mask is None:
        return [1]
    
    result = []
    for x in range(N):
        if (best_overall_mask >> x) & 1:
            result.append(x + 1)
    
    return sorted(result)
