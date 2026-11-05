import time
import random

def construct():
    N = 256
    # Bit i represents element i+1.
    # We use a large integer to represent the subset.
    
    # Precompute 4-AP constraints
    # four_mask[x] is a bitmask of elements y such that if x is in the set, 
    # and the other elements of the 4-AP containing x and y are in the set, it's invalid.
    # Actually, it's easier to store for each x, a list of "forbidden patterns" or just check on the fly with precomputed bitsets.
    
    # Let's precompute for each x:
    # four_conflicts[x]: A list of integers, each integer is a bitmask of the OTHER 3 elements in a 4-AP containing x.
    # If (current_mask & mask) == mask, then adding x would complete a 4-AP (or x is already there and completes it).
    # Wait, if we are adding x, we need to check if the other 3 are present.
    # If we are checking if a set is valid, we check if any x in the set has its 3 partners in the set.
    
    # Better approach for SA:
    # Precompute `four_mask[x]` as a list of bitmasks (ints). Each int represents the set of other 3 elements.
    # Precompute `three_mask[x]` as a list of bitmasks (ints). Each int represents the set of other 2 elements.
    
    four_masks = [[] for _ in range(N+1)]
    three_masks = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for d in range(1, N+1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e > N: break
            
            # 4-AP: a, b, c, e
            # For a: others b, c, e
            mask_a = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_masks[a].append(mask_a)
            
            # For b: others a, c, e
            mask_b = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            four_masks[b].append(mask_b)
            
            # For c: others a, b, e
            mask_c = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            four_masks[c].append(mask_c)
            
            # For e: others a, b, c
            mask_e = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            four_masks[e].append(mask_e)
            
            # 3-APs from this 4-AP? No, 3-APs are a, a+d, a+2d.
            # We should generate all 3-APs separately to be safe, or just use the ones from 4-AP generation?
            # A 3-AP is (a, a+d, a+2d).
            # The 4-AP loop generates a, a+d, a+2d, a+3d.
            # The 3-APs contained in this are (a, a+d, a+2d) and (a+d, a+2d, a+3d).
            # But there are 3-APs not part of a 4-AP.
            # So we must generate all 3-APs independently.
            
    # Generate all 3-APs
    for a in range(1, N+1):
        for d in range(1, N+1):
            b = a + d
            c = a + 2*d
            if c > N: break
            
            # 3-AP: a, b, c
            mask_a = (1 << (b-1)) | (1 << (c-1))
            three_masks[a].append(mask_a)
            
            mask_b = (1 << (a-1)) | (1 << (c-1))
            three_masks[b].append(mask_b)
            
            mask_c = (1 << (a-1)) | (1 << (b-1))
            three_masks[c].append(mask_c)

    # Helper: Check if adding x to mask creates a 4-AP
    def is_valid_add(mask, x):
        for m in four_masks[x]:
            if mask & m == m:
                return False
        return True

    # Helper: Count 3-APs gained by adding x
    def gain_add(mask, x):
        gain = 0
        for m in three_masks[x]:
            if mask & m == m:
                gain += 1
        return gain

    # Helper: Count 3-APs lost by removing x
    def loss_remove(mask, x):
        # x is in mask
        loss = 0
        for m in three_masks[x]:
            # m contains the other 2. If they are in mask, the AP exists.
            # Removing x destroys this AP.
            if mask & m == m:
                loss += 1
        return loss

    # Initialization: Greedy
    # Try to build a large set with many 3-APs.
    # Randomized greedy: pick element with highest gain that is valid.
    # To speed up, we can just do a simple random greedy a few times.
    
    best_mask = 0
    best_score = 0
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    # Quick Greedy Init
    for _ in range(20):
        current_mask = 0
        current_score = 0
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_add(current_mask, x):
                g = gain_add(current_mask, x)
                current_mask |= (1 << (x-1))
                current_score += g
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        if time.time() - start_time > 2.0:
            break

    current_mask = best_mask
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    # Cooling rate needs to be chosen such that we do enough steps.
    # Let's fix number of steps based on time? No, just loop until time limit.
    # To avoid slow cooling, we can use a fixed step count or check time.
    
    while time.time() - start_time < time_limit:
        # Move: Add, Remove, or Swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add
            x = random.randint(1, N)
            if not (current_mask & (1 << (x-1))):
                if is_valid_add(current_mask, x):
                    g = gain_add(current_mask, x)
                    new_mask = current_mask | (1 << (x-1))
                    new_score = current_score + g
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.71828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                            
        elif move_type < 0.6:
            # Remove
            # Pick random x in mask
            # Efficient: pick random bit
            # Simple: random x, check if in mask
            x = random.randint(1, N)
            if current_mask & (1 << (x-1)):
                l = loss_remove(current_mask, x)
                new_mask = current_mask & ~(1 << (x-1))
                new_score = current_score - l
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.71828, delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
        else:
            # Swap
            u = random.randint(1, N)
            v = random.randint(1, N)
            if u != v and (current_mask & (1 << (u-1))) and not (current_mask & (1 << (v-1))):
                mask_without_u = current_mask & ~(1 << (u-1))
                if is_valid_add(mask_without_u, v):
                    l = loss_remove(current_mask, u)
                    g = gain_add(mask_without_u, v)
                    delta = g - l
                    new_mask = mask_without_u | (1 << (v-1))
                    new_score = current_score + delta
                    if delta > 0 or random.random() < pow(2.71828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        
        T *= 0.999
        if T < T_min:
            T = T_min

    # Extract result
    result = []
    for i in range(1, N+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return sorted(result)
