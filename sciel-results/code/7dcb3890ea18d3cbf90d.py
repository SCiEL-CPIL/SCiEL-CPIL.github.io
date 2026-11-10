def construct():
    import time
    import random
    from math import gcd
    from itertools import combinations

    deadline = time.time() + 85

    n = 128

    # Precompute GCD table
    gcd_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute bad_pairs_for_x[x] = set of pairs (a,b) with a<b
    # such that (a,b,x) is a bad triple (gcd(a,b)==gcd(a,x)==gcd(b,x))
    # We only need to check pairs where a,b < x or a,b > x? No, any pair.
    # But for efficiency, let's store pairs as tuples (min, max).
    bad_pairs_for_x = [set() for _ in range(n+1)]
    
    # Optimization: Iterate over all triples (a,b,c) with a<b<c
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            for c in range(b+1, n+1):
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    bad_pairs_for_x[c].add((a,b))
                    bad_pairs_for_x[b].add((a,c))
                    bad_pairs_for_x[a].add((b,c))

    # Fast check: can x be added to S?
    # We need to check if any pair (a,b) in bad_pairs_for_x[x] is fully contained in S.
    # To make this O(1) or O(log n) per pair, we can maintain a bitmask of S.
    # n=128, so we can use a Python int as a bitmask (128 bits).
    # bit i is set if (i+1) is in S.
    
    def can_add_bitmask(x, mask):
        # Check if any pair (a,b) in bad_pairs_for_x[x] has both bits set in mask
        # Pair (a,b) corresponds to bits (a-1) and (b-1)
        for a, b in bad_pairs_for_x[x]:
            if (mask & (1 << (a-1))) and (mask & (1 << (b-1))):
                return False
        return True

    def greedy_fill(mask, order):
        """Fill mask greedily following the given order."""
        for x in order:
            if not (mask & (1 << (x-1))):
                if can_add_bitmask(x, mask):
                    mask |= (1 << (x-1))
        return mask

    def popcount(mask):
        return bin(mask).count('1')

    def mask_to_list(mask):
        return [i+1 for i in range(n) if (mask & (1 << i))]

    # Local Search Strategy:
    # 1. Start with a greedy solution.
    # 2. Try to improve by:
    #    a. Adding any element that fits (greedy fill).
    #    b. Swapping an element out and a different one in if it increases size.
    #    c. Removing a few elements and re-filling (kick).
    
    def local_search(initial_mask, time_limit):
        current_mask = initial_mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        
        start_time = time.time()
        rng = random.Random(42)
        
        while time.time() < time_limit:
            improved = False
            
            # Phase 1: Greedy Add
            # Try to add any element not in current_mask
            for x in range(1, n+1):
                if not (current_mask & (1 << (x-1))):
                    if can_add_bitmask(x, current_mask):
                        current_mask |= (1 << (x-1))
                        current_size += 1
                        improved = True
            
            if current_size > best_size:
                best_mask = current_mask
                best_size = current_size
                
            if not improved:
                # Phase 2: Swap Search
                # Try swapping one element out and one in
                # To be efficient, we only try swaps if we have time.
                # We iterate over elements in current_mask to remove,
                # and elements not in current_mask to add.
                # This is O(|S| * |U\S| * Cost(can_add)).
                # Cost(can_add) is O(|bad_pairs|).
                
                # Optimization: Precompute list of elements in set and not in set
                in_set = [i+1 for i in range(n) if (current_mask & (1 << i))]
                not_in_set = [i+1 for i in range(n) if not (current_mask & (1 << i))]
                
                found_swap = False
                # Randomize order to avoid bias
                rng.shuffle(in_set)
                rng.shuffle(not_in_set)
                
                for out_x in in_set:
                    if time.time() > time_limit:
                        break
                    # Remove out_x
                    temp_mask = current_mask & ~(1 << (out_x-1))
                    
                    for in_x in not_in_set:
                        if time.time() > time_limit:
                            break
                        # Check if in_x can be added to temp_mask
                        if can_add_bitmask(in_x, temp_mask):
                            # Swap is valid. Size remains same.
                            # But we want to increase size. A swap alone doesn't increase size.
                            # However, after a swap, we might be able to add MORE elements.
                            # So we perform the swap, then try to greedy add again.
                            new_mask = temp_mask | (1 << (in_x-1))
                            new_size = current_size # Size unchanged by swap itself
                            
                            # Try to fill after swap
                            for y in range(1, n+1):
                                if not (new_mask & (1 << (y-1))):
                                    if can_add_bitmask(y, new_mask):
                                        new_mask |= (1 << (y-1))
                                        new_size += 1
                            
                            if new_size > best_size:
                                best_mask = new_mask
                                best_size = new_size
                                current_mask = new_mask
                                current_size = new_size
                                improved = True
                                found_swap = True
                                break
                    if found_swap:
                        break
                
                if found_swap:
                    continue # Restart loop to try more adds
            
            # Phase 3: Kick (if stuck)
            # Remove k random elements and re-greedy
            if current_size < best_size or not improved:
                # If we haven't improved in this iteration, do a kick
                # But only if we have time
                if time.time() < time_limit - 1:
                    k = rng.randint(2, 4)
                    if len(in_set) > k:
                        to_remove = rng.sample(in_set, k)
                        kick_mask = current_mask
                        for x in to_remove:
                            kick_mask &= ~(1 << (x-1))
                        
                        # Re-greedy fill
                        kick_mask = greedy_fill(kick_mask, list(range(1, n+1)))
                        kick_size = popcount(kick_mask)
                        
                        if kick_size > best_size:
                            best_mask = kick_mask
                            best_size = kick_size
                            current_mask = kick_mask
                            current_size = kick_size
                            improved = True

        return best_mask, best_size

    # Main Loop: Multi-start
    rng = random.Random(123)
    global_best_mask = 0
    global_best_size = 0
    
    # Initial greedy starts with different orders
    for i in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n+1))
        rng.shuffle(order)
        mask = greedy_fill(0, order)
        size = popcount(mask)
        if size > global_best_size:
            global_best_mask = mask
            global_best_size = size
            
        # Local search from this start
        # Allocate time proportional to remaining budget
        remaining = deadline - time.time()
        if remaining < 2:
            break
        ls_time = time.time() + min(remaining * 0.8, 5.0) # Cap each LS to 5s initially
        best_mask, best_size = local_search(mask, ls_time)
        
        if best_size > global_best_size:
            global_best_mask = best_mask
            global_best_size = best_size

    # If we have time left, do a final intensive local search from the global best
    remaining = deadline - time.time()
    if remaining > 5:
        # Run local search again from the current best, with more time
        final_mask, final_size = local_search(global_best_mask, deadline - 1)
        if final_size > global_best_size:
            global_best_mask = final_mask
            global_best_size = final_size

    return mask_to_list(global_best_mask)
