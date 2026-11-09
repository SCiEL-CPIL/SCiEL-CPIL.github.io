"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # Pre-compute all 4-element subsets of [1,16] as bitmasks for speed
    # and also keep the original tuples for output
    all_sets_info = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i-1))
        all_sets_info.append((mask, row))
    
    n_total = len(all_sets_info)
    all_masks = [info[0] for info in all_sets_info]
    all_rows = [info[1] for info in all_sets_info]
    
    # Helper: check if mask can be added to a set of chosen masks
    def can_add_mask(mask, chosen_masks):
        n = len(chosen_masks)
        for i in range(n):
            ma = chosen_masks[i]
            for j in range(i + 1, n):
                mb = chosen_masks[j]
                # Sunflower condition: (ma & mb) == (ma & mask) == (mb & mask)
                if (ma & mb) == (ma & mask) == (mb & mask):
                    return False
        return True
    
    # Helper: run greedy given an order of indices
    def greedy_pass(order):
        chosen_masks = []
        chosen_indices = []
        for idx in order:
            mask = all_masks[idx]
            if can_add_mask(mask, chosen_masks):
                chosen_masks.append(mask)
                chosen_indices.append(idx)
        return chosen_indices
    
    # Local search: try to improve by swapping one chosen set with one unchosen set
    def local_search(chosen_indices, max_iters=200):
        chosen_set = set(chosen_indices)
        chosen_masks = [all_masks[i] for i in chosen_indices]
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > deadline - 5:
                break
            
            # Try each swap: remove one chosen, add one unchosen
            for out_idx in range(len(chosen_indices)):
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove out_idx from chosen_masks for checking
                # We need to check if in_mask can be added to chosen_masks without out_mask
                out_mask = chosen_masks[out_idx]
                
                # To avoid O(N) removal every time, we can just pass a modified list or logic
                # But N is small (~250), so creating a temp list is acceptable for the inner loop
                # However, we are iterating over unchosen (up to 1820) for each out_idx (up to 250).
                # That's 450k checks per iteration. Each check is O(K^2) ~ 30k ops.
                # 450k * 30k = 13.5 billion ops. This is too slow.
                
                # Optimization: Pre-compute "conflicts" or use a faster check?
                # Or limit the search.
                
                # Let's optimize the check:
                # Instead of rebuilding temp_chosen_masks, we can just skip out_idx in the loop.
                
                found_swap = False
                for in_idx in unchosen:
                    in_mask = all_masks[in_idx]
                    
                    # Check if in_mask is compatible with all chosen masks except out_mask
                    compatible = True
                    for i in range(len(chosen_masks)):
                        if i == out_idx:
                            continue
                        ma = chosen_masks[i]
                        # We need to check pairs (ma, mb) where mb is also chosen and != out_idx
                        # This logic is getting complex to inline efficiently.
                        
                        # Let's stick to a simpler but slightly slower local search or limit iterations.
                        # Actually, let's just use the can_add_mask logic but on a filtered list.
                        # Since K is small, let's just build the filtered list once per out_idx?
                        # No, that's expensive.
                        
                        # Alternative: Just check pairs involving in_mask.
                        # (ma & mb) == (ma & in_mask) == (mb & in_mask)
                        # We need to check all pairs (ma, mb) from chosen \ {out_mask}.
                        
                        pass
                    
                    # This inner loop is too complex to optimize on the fly without restructuring.
                    # Let's use a simpler local search: random swaps.
                    break # Break inner loop to restart with a different strategy if needed
                    
                # Let's restart the local search with a simpler approach:
                # Try to swap a random chosen element with a random unchosen one.
                # This is less systematic but faster per step.
                
                # Actually, let's just use the systematic swap but limit the number of unchosen candidates tested per out_idx.
                # Or just do a few iterations of this.
                
                # Given the time constraints, let's simplify local_search to just try a limited number of random swaps.
                # This avoids the heavy O(K^2 * N_unchosen) per step.
                
                # Revised Local Search: Random Swap
                if random.random() < 0.5: # Probability to do a random swap instead of systematic
                    out_idx = random.randint(0, len(chosen_indices) - 1)
                    in_idx = random.choice(unchosen)
                    
                    # Check if swap is valid
                    temp_masks = [m for k, m in enumerate(chosen_masks) if k != out_idx]
                    if can_add_mask(all_masks[in_idx], temp_masks):
                        # Perform swap
                        chosen_indices[out_idx] = in_idx
                        chosen_masks[out_idx] = all_masks[in_idx]
                        
                        # Update sets
                        old_idx = chosen_indices[out_idx] # This is wrong, we already updated index
                        # Let's track indices properly
                        # chosen_indices[out_idx] was the old index
                        # We need to update chosen_set and unchosen
                        old_idx = chosen_indices[out_idx] # Wait, I overwrote it?
                        # Let's re-implement this part carefully.
                        
                        pass
                
                break # Stop systematic loop, rely on random swaps in next iteration?
            
            # Let's implement a clean Random Swap Local Search instead of the systematic one above.
            # It's much easier to implement correctly and fast enough.
            
        return chosen_indices

    # Revised Local Search: Random Swaps
    def local_search_random(chosen_indices, max_iters=500):
        chosen_set = set(chosen_indices)
        chosen_masks = [all_masks[i] for i in chosen_indices]
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            if not unchosen:
                break
                
            # Pick a random chosen index to remove
            out_pos = random.randint(0, len(chosen_indices) - 1)
            # Pick a random unchosen index to add
            in_pos = random.randint(0, len(unchosen) - 1)
            
            out_idx = chosen_indices[out_pos]
            in_idx = unchosen[in_pos]
            
            # Check if in_idx can replace out_idx
            # Remove out_idx from chosen_masks temporarily
            temp_masks = []
            for k in range(len(chosen_masks)):
                if k != out_pos:
                    temp_masks.append(chosen_masks[k])
            
            if can_add_mask(all_masks[in_idx], temp_masks):
                # Perform swap
                chosen_indices[out_pos] = in_idx
                chosen_masks[out_pos] = all_masks[in_idx]
                
                # Update tracking sets
                chosen_set.discard(out_idx)
                chosen_set.add(in_idx)
                
                # Update unchosen list
                unchosen[in_pos] = out_idx
                
        return chosen_indices

    best_family = []
    best_size = 0
    
    # Define several deterministic orderings
    indices = list(range(n_total))
    
    # 1. Lexicographic (default)
    order_1 = indices[:]
    
    # 2. Reverse Lexicographic
    order_2 = indices[::-1]
    
    # 3. Sorted by bitmask value
    order_3 = sorted(indices, key=lambda i: all_masks[i])
    
    # 4. Sorted by sum of elements
    order_4 = sorted(indices, key=lambda i: sum(all_rows[i]))
    
    orders = [order_1, order_2, order_3, order_4]
    
    # Add a couple of random orders
    for _ in range(3):
        order_r = indices[:]
        random.shuffle(order_r)
        orders.append(order_r)
    
    for order in orders:
        if time.time() > deadline:
            break
        
        # Greedy Pass
        family_indices = greedy_pass(order)
        
        # Local Search
        if time.time() < deadline - 10: # Reserve time for local search
            family_indices = local_search_random(family_indices, max_iters=300)
        
        current_size = len(family_indices)
        if current_size > best_size:
            best_size = current_size
            best_family = family_indices[:]
            
        # If we hit the cap, we can stop
        if best_size >= 250:
            break
            
    # Convert indices back to sorted lists of elements
    return [sorted(all_rows[i]) for i in best_family]
# EVOLVE-BLOCK-END
