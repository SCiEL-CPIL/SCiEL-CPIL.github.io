def construct():
    import time
    import random

    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit

    N = 512

    # Precompute conflict masks for fast can_add
    # ap_masks[x] is a list of masks. Each mask represents the other 3 elements in a 4-AP containing x.
    # If (S_bits & mask) == mask, then x cannot be added.
    ap_masks = [[] for _ in range(N + 1)]
    
    # Iterate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            
            # AP: a, b, c, e
            # For a: others are b, c, e
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            ap_masks[a].append(mask_a)
            
            # For b: others are a, c, e
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            ap_masks[b].append(mask_b)
            
            # For c: others are a, b, e
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            ap_masks[c].append(mask_c)
            
            # For e: others are a, b, c
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            ap_masks[e].append(mask_e)

    def can_add_fast(S_bits, x):
        idx = x - 1
        if (S_bits >> idx) & 1:
            return False
        for mask in ap_masks[x]:
            if (S_bits & mask) == mask:
                return False
        return True

    # Generate Base-3 (Salem-Spencer) set
    # Numbers in [1, N] with base-3 digits only 0 or 1.
    def generate_base3_set():
        S_bits = 0
        count = 0
        # Max power of 3 <= N
        # 3^6 = 729 > 512, so up to 3^5
        for b5 in [0, 1]:
            for b4 in [0, 1]:
                for b3 in [0, 1]:
                    for b2 in [0, 1]:
                        for b1 in [0, 1]:
                            for b0 in [0, 1]:
                                val = b5 * 243 + b4 * 81 + b3 * 27 + b2 * 9 + b1 * 3 + b0
                                if 1 <= val <= N:
                                    S_bits |= (1 << (val - 1))
                                    count += 1
        return S_bits, count

    # Initial states
    # 1. Base-3 set
    S_base, count_base = generate_base3_set()
    
    # 2. Random Greedy (a few runs to see if we can beat base-3 or find a different basin)
    best_S_bits = S_base
    best_count = count_base
    
    for i in range(20):
        if time.time() > deadline - 5.0: break
        random.seed(i)
        S_bits = 0
        count = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add_fast(S_bits, x):
                S_bits |= (1 << (x - 1))
                count += 1
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Simulated Annealing
    S_bits = best_S_bits
    count = best_count
    S_list = [i + 1 for i in range(N) if (S_bits >> i) & 1]
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    all_elements = list(range(1, N + 1))
    
    # SA Loop
    while T > min_T and time.time() < deadline - 2.0:
        # Move types:
        # 1. Try to add a random x (improvement)
        # 2. Swap: remove x, add y (neutral or improvement if we consider future adds, but usually neutral in size)
        #    To allow escaping local minima, we might need to accept size reductions or use a "kick".
        #    However, simple SA with swap (size constant) and add (size +1) is standard.
        #    The issue is getting stuck where no add is valid and no swap helps.
        #    We can introduce a "remove" move with low probability to allow the set to shrink temporarily?
        #    Or, better: "Remove x, Add y" where y is chosen to maximize future potential?
        #    Let's stick to standard SA:
        #    - Add move: +1 size.
        #    - Swap move: 0 size change.
        #    - (Optional) Remove move: -1 size change, accepted with prob exp(-1/T).
        
        move_type = random.random()
        if move_type < 0.5:
            # Try to add
            y = random.choice(all_elements)
            if can_add_fast(S_bits, y):
                S_bits |= (1 << (y - 1))
                S_list.append(y)
                count += 1
                if count > best_count:
                    best_count = count
                    best_S_bits = S_bits
        elif move_type < 0.9:
            # Swap
            if not S_list: continue
            x = random.choice(S_list)
            y = random.choice(all_elements)
            if x == y: continue
            
            # Remove x
            S_bits ^= (1 << (x - 1))
            S_list.remove(x)
            
            # Try to add y
            if can_add_fast(S_bits, y):
                S_bits |= (1 << (y - 1))
                S_list.append(y)
                # Size unchanged. Accept always? 
                # In SA, we accept based on energy. Here energy = -count.
                # Delta E = 0. Accept with prob 1.
                # But this is a random walk on the manifold of valid sets of same size.
                # It helps to find a configuration where an add becomes possible.
            else:
                # Revert
                S_bits |= (1 << (x - 1))
                S_list.append(x)
        else:
            # Remove (kick)
            if S_list:
                x = random.choice(S_list)
                S_bits ^= (1 << (x - 1))
                S_list.remove(x)
                count -= 1
                # Accept with prob exp(1/T) ? No, Delta E = +1 (worse).
                # Accept if random() < exp(-1/T).
                if random.random() < (2.71828 ** (-1.0 / T)):
                    # Keep removed
                    pass
                else:
                    # Revert
                    S_bits |= (1 << (x - 1))
                    S_list.append(x)
                    count += 1

        T *= cooling

    # Final Polish: Greedy add
    for x in all_elements:
        if time.time() > deadline - 1.0: break
        if can_add_fast(best_S_bits, x):
            best_S_bits |= (1 << (x - 1))
            best_count += 1

    result = [i + 1 for i in range(N) if (best_S_bits >> i) & 1]
    return sorted(result)
