import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85.0 # Safety margin for 90s limit
    
    # Precomputation
    thre_aps = []
    four_aps = []
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            thre_aps.append((a, b, c))
            
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps.append((a, b, c, e))
            
    elem_3_aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(thre_aps):
        mask_bc = (1 << (b-1)) | (1 << (c-1))
        mask_ac = (1 << (a-1)) | (1 << (c-1))
        mask_ab = (1 << (a-1)) | (1 << (b-1))
        
        elem_3_aps[a].append(mask_bc)
        elem_3_aps[b].append(mask_ac)
        elem_3_aps[c].append(mask_ab)
        
    elem_4_aps = [[] for _ in range(N + 1)]
    for idx, (a, b, c, d) in enumerate(four_aps):
        mask_bcd = (1 << (b-1)) | (1 << (c-1)) | (1 << (d-1))
        mask_acd = (1 << (a-1)) | (1 << (c-1)) | (1 << (d-1))
        mask_abd = (1 << (a-1)) | (1 << (b-1)) | (1 << (d-1))
        mask_abc = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
        
        elem_4_aps[a].append(mask_bcd)
        elem_4_aps[b].append(mask_acd)
        elem_4_aps[c].append(mask_abd)
        elem_4_aps[d].append(mask_abc)

    def count_3aps(mask):
        count = 0
        for idx, (a, b, c) in enumerate(thre_aps):
            if (mask & (1 << (a-1))) and (mask & (1 << (b-1))) and (mask & (1 << (c-1))):
                count += 1
        return count

    def can_add(mask, x):
        for other_mask in elem_4_aps[x]:
            if (mask & other_mask) == other_mask:
                return False
        return True

    def gain_3aps(mask, x):
        gain = 0
        for other_mask in elem_3_aps[x]:
            if (mask & other_mask) == other_mask:
                gain += 1
        return gain

    def loss_3aps(mask, x):
        loss = 0
        for other_mask in elem_3_aps[x]:
            if (mask & other_mask) == other_mask:
                loss += 1
        return loss

    # Initialization: Greedy
    mask = 0
    for x in range(1, N + 1):
        if can_add(mask, x):
            mask |= (1 << (x-1))
            
    current_score = count_3aps(mask)
    
    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try all single additions
        for x in range(1, N + 1):
            if not (mask & (1 << (x-1))):
                if can_add(mask, x):
                    gain = gain_3aps(mask, x)
                    if gain > 0:
                        mask |= (1 << (x-1))
                        current_score += gain
                        improved = True
        
        if improved:
            continue
            
        # 2. Try swaps
        # Limit iterations to avoid getting stuck in a long loop of bad swaps
        stuck_count = 0
        max_stuck = 100
        
        while stuck_count < max_stuck and time.time() - start_time < time_limit:
            # Get list of elements in mask
            bits = []
            temp = mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                bits.append(idx + 1)
                temp ^= lsb
                
            if not bits:
                break
                
            x = random.choice(bits)
            
            # Pick random y not in mask
            y = random.randint(1, N)
            while (mask & (1 << (y-1))):
                y = random.randint(1, N)
                
            mask_without_x = mask ^ (1 << (x-1))
            
            # Check validity
            valid = True
            for other_mask in elem_4_aps[y]:
                if (mask_without_x & other_mask) == other_mask:
                    valid = False
                    break
                    
            if not valid:
                stuck_count += 1
                continue
                
            # Check score
            loss = loss_3aps(mask, x)
            gain = gain_3aps(mask_without_x, y)
            
            if gain > loss:
                mask = mask_without_x | (1 << (y-1))
                current_score += (gain - loss)
                improved = True
                break
            else:
                stuck_count += 1
                
        if not improved:
            # Perturbation
            # Remove 5 random elements
            bits = []
            temp = mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                bits.append(idx + 1)
                temp ^= lsb
                
            if bits:
                to_remove = random.sample(bits, min(5, len(bits)))
                for x in to_remove:
                    mask ^= (1 << (x-1))
                current_score = count_3aps(mask)
                # We don't set improved=True here, so the loop continues
                # But we want to restart the "try additions" phase
                # So we just break out of the "stuck" handling and go to next while iteration
                # The next iteration will try additions again

    result = []
    temp = mask
    while temp:
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        temp ^= lsb
        
    return sorted(result)
