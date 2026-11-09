import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256
    
    # Precompute masks for 4-AP conflicts and 3-AP gains
    conflict_masks = [[] for _ in range(N + 1)]
    gain_masks = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            mask_a = (1 << b) | (1 << c) | (1 << e)
            mask_b = (1 << a) | (1 << c) | (1 << e)
            mask_c = (1 << a) | (1 << b) | (1 << e)
            mask_e = (1 << a) | (1 << b) | (1 << c)
            
            conflict_masks[a].append(mask_a)
            conflict_masks[b].append(mask_b)
            conflict_masks[c].append(mask_c)
            conflict_masks[e].append(mask_e)
            
    # Generate all 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > N:
                break
            
            mask_a = (1 << b) | (1 << c)
            mask_b = (1 << a) | (1 << c)
            mask_c = (1 << a) | (1 << b)
            
            gain_masks[a].append(mask_a)
            gain_masks[b].append(mask_b)
            gain_masks[c].append(mask_c)

    def is_valid_add(x, M):
        for mask in conflict_masks[x]:
            if (M & mask) == mask:
                return False
        return True

    def count_3ap_gain(x, M):
        gain = 0
        for mask in gain_masks[x]:
            if (M & mask) == mask:
                gain += 1
        return gain

    def count_total_3ap(M):
        total = 0
        m = M
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            total += count_3ap_gain(x, M)
            m ^= lsb
        return total // 3

    def local_search_first_improvement(M, current_score):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try single additions (First Improvement)
            # We iterate x from 1 to N. If valid and gain > 0, add and restart? 
            # Or just add and continue? Adding x might invalidate y if x,y in 4AP.
            # To be safe and simple, we scan 1..N. If we add x, we set improved=True and break to restart the loop?
            # Restarting ensures we don't miss cascading additions.
            for x in range(1, N + 1):
                if not (M & (1 << x)):
                    if is_valid_add(x, M):
                        gain = count_3ap_gain(x, M)
                        if gain > 0:
                            M |= (1 << x)
                            current_score += gain
                            improved = True
                            break # Restart the while loop to re-evaluate all additions
                
            if time.time() > deadline - 5:
                break

            # 2. Try swaps (First Improvement)
            A = []
            m = M
            while m:
                lsb = m & -m
                A.append(lsb.bit_length() - 1)
                m ^= lsb
                
            B = [x for x in range(1, N + 1) if not (M & (1 << x))]
            
            # To make this faster, we can precompute loss for each a in A
            # loss_a = count_3ap_gain(a, M)
            # But M changes if we swap.
            
            found_swap = False
            # Shuffle A and B to randomize the order of search, helping escape local minima
            random.shuffle(A)
            random.shuffle(B)
            
            for a in A:
                if time.time() > deadline - 5:
                    break
                M_temp = M & ~(1 << a)
                loss_a = count_3ap_gain(a, M) # 3-APs containing a in current M
                
                for b in B:
                    if not is_valid_add(b, M_temp):
                        continue
                    gain_b = count_3ap_gain(b, M_temp)
                    delta = gain_b - loss_a
                    if delta > 0:
                        # Perform swap
                        M = M_temp | (1 << b)
                        current_score += delta
                        improved = True
                        found_swap = True
                        break # Break inner loop, restart outer while loop
                if found_swap:
                    break
                    
        return M, current_score

    best_M = 0
    best_score = 0
    
    # Initial Greedy Construction
    # Use more iterations for initialization
    for _ in range(20):
        if time.time() > deadline - 15:
            break
        M = 0
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        while remaining and time.time() < deadline - 15:
            best_x = -1
            best_gain = -1
            # Randomized greedy: pick best among top 3? Or just best.
            # Let's stick to best for stability, but random shuffle of remaining helps.
            for x in remaining:
                if is_valid_add(x, M):
                    g = count_3ap_gain(x, M)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
            if best_x == -1:
                break
            M |= (1 << best_x)
            remaining.remove(best_x)
        
        M, score = local_search_first_improvement(M, count_total_3ap(M))
        if score > best_score:
            best_score = score
            best_M = M

    # Iterated Local Search
    M = best_M
    current_score = best_score
    
    while time.time() < deadline - 2:
        # Local Search
        M, current_score = local_search_first_improvement(M, current_score)
        
        if current_score > best_score:
            best_score = current_score
            best_M = M
            
        # Perturbation
        A = []
        m = M
        while m:
            lsb = m & -m
            A.append(lsb.bit_length() - 1)
            m ^= lsb
            
        if len(A) > 5:
            # Random kick: remove k elements
            k = random.randint(3, 8)
            to_remove = random.sample(A, k)
            for x in to_remove:
                M &= ~(1 << x)
            current_score = count_total_3ap(M)
        else:
            # Restart if too small
            M = 0
            current_score = 0
            remaining = list(range(1, N + 1))
            random.shuffle(remaining)
            while remaining and time.time() < deadline - 2:
                best_x = -1
                best_gain = -1
                for x in remaining:
                    if is_valid_add(x, M):
                        g = count_3ap_gain(x, M)
                        if g > best_gain:
                            best_gain = g
                            best_x = x
                if best_x == -1:
                    break
                M |= (1 << best_x)
                remaining.remove(best_x)
            current_score = count_total_3ap(M)

    # Convert bitmask to sorted list
    result = []
    m = best_M
    while m:
        lsb = m & -m
        result.append(lsb.bit_length() - 1)
        m ^= lsb
        
    return sorted(result)
