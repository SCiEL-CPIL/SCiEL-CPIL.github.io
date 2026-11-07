import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 60s, but we want to return well before.
    deadline = start_time + 50.0
    
    n = P['n']
    max_val = n
    full_mask = (1 << n) - 1
    
    # Precompute pairs for sums
    # pairs_sum[s] = list of (a, b) such that a + b = s, a < b
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))
                
    def get_bits(mask):
        """Yield indices of set bits."""
        m = mask
        while m:
            lsb = m & -m
            yield lsb.bit_length() - 1
            m ^= lsb

    def is_valid_addition(x, mask):
        """
        Check if adding x to valid set 'mask' results in a valid set.
        x is 1-indexed. mask is bitset where bit i corresponds to value i+1.
        """
        if mask & (1 << (x - 1)):
            return True # Already in set
            
        x_bit = 1 << (x - 1)
        
        # Case 1: x is one of the members a,b,c.
        # WLOG a=x. Need b,c in S such that b+c in S, x+b in S, x+c in S.
        # Let A = {b in S | x+b in S}
        # We need to check if there exist distinct b, c in A such that b+c in S.
        
        # Efficiently find A
        # b is in A if bit (b-1) is set in mask AND bit (x+b-1) is set in mask.
        # x+b ranges from x+1 to x+128.
        # We can just iterate b in S.
        
        A_bits = 0
        # Iterate through all b in S
        m = mask
        while m:
            lsb = m & -m
            b_idx = lsb.bit_length() - 1
            b_val = b_idx + 1
            # Check if x + b_val is in S
            sum_val = x + b_val
            if sum_val <= max_val:
                if mask & (1 << (sum_val - 1)):
                    A_bits |= lsb
            m ^= lsb
            
        # Now check if there are b, c in A_bits, b != c, such that b+c in S
        # Iterate b in A_bits
        m = A_bits
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            
            # We need c in A_bits, c != b, such that b+c in S
            # Let C_b = {c in S | b+c in S}
            # We need A_bits & C_b to have a bit set that is not b.
            
            # Iterate c in S to build C_b? No, just check intersection.
            # C_b bits: c is in C_b if c in S and b+c in S.
            # We can compute C_b on the fly or just check pairs.
            # Given small N, iterating c in S for each b in A is O(N^2) worst case.
            # But A is usually small.
            
            found_conflict = False
            m_c = mask
            while m_c:
                lsb_c = m_c & -m_c
                c_idx = lsb_c.bit_length() - 1
                c_val = c_idx + 1
                m_c ^= lsb_c
                
                if c_idx == b_idx:
                    continue
                
                # Check if c is in A_bits
                if not (A_bits & lsb_c):
                    continue
                    
                # Check if b+c is in S
                sum_val = b_val + c_val
                if sum_val <= max_val and (mask & (1 << (sum_val - 1))):
                    found_conflict = True
                    break
            
            if found_conflict:
                return False
        
        # Case 2: x is one of the sums.
        # WLOG a+b = x. a,b in S.
        # Need c in S, c != a, c != b, such that a+c in S and b+c in S.
        # Note: a+c=x implies c=b (excluded). b+c=x implies c=a (excluded).
        # So we just need a+c in S and b+c in S.
        
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
                
            # Found a,b in S summing to x.
            # Check for c
            # We need c in S \ {a,b} such that (a+c in S) and (b+c in S)
            
            # Let U_a = {c in S | a+c in S}
            # Let U_b = {c in S | b+c in S}
            # Check if (U_a & U_b) has any bit other than a or b.
            
            # Build U_a and U_b bits? Or just iterate c?
            # Iterating c in S is O(N).
            
            conflict = False
            m_c = mask
            while m_c:
                lsb_c = m_c & -m_c
                c_idx = lsb_c.bit_length() - 1
                c_val = c_idx + 1
                m_c ^= lsb_c
                
                if c_idx == (a - 1) or c_idx == (b - 1):
                    continue
                
                # Check if a+c in S
                sum_ac = a + c_val
                if sum_ac > max_val or not (mask & (1 << (sum_ac - 1))):
                    continue
                    
                # Check if b+c in S
                sum_bc = b + c_val
                if sum_bc > max_val or not (mask & (1 << (sum_bc - 1))):
                    continue
                
                conflict = True
                break
            
            if conflict:
                return False
                
        return True

    def try_add(mask, x):
        if is_valid_addition(x, mask):
            return mask | (1 << (x - 1))
        return mask

    def try_remove(mask, x):
        # Removing an element from a valid set always results in a valid set.
        return mask & ~(1 << (x - 1))

    # Initial valid set: Upper half
    best_mask = 0
    for i in range(n//2, n):
        best_mask |= (1 << i)
    best_size = n - n//2
    
    current_mask = best_mask
    current_size = best_size
    
    # Local Search
    # Strategy: Try to add random elements. If stuck, remove one and try to add two.
    # Or Simulated Annealing: Accept worse moves with prob exp(-delta/T).
    # Since we only care about size, we prefer larger sets.
    # Moves:
    # 1. Add x (size +1) -> Always accept if valid.
    # 2. Remove x (size -1) -> Only accept if it allows adding 2+ later? 
    #    Better: 1-out, 2-in swap.
    
    # To keep it fast, we'll do:
    # Loop:
    #   Try to add a random unused element.
    #   If successful, continue.
    #   If failed, try a 1-out, 2-in move.
    #   If failed, random restart or perturbation.
    
    unused_elements = [i for i in range(1, n + 1) if not (current_mask & (1 << (i-1)))]
    current_elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i-1)))]
    
    while time.time() < deadline:
        # Try greedy add
        if unused_elements:
            random.shuffle(unused_elements)
            added = False
            for x in unused_elements:
                new_mask = try_add(current_mask, x)
                if new_mask != current_mask:
                    # Success
                    current_mask = new_mask
                    current_size += 1
                    unused_elements.remove(x)
                    current_elements.append(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    break
            if added:
                continue
        
        # If we couldn't add, try 1-out, 2-in
        # Remove a random element, then try to add two
        if current_elements:
            # Pick a random element to remove
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            # Try to add two elements from unused
            # We need to find two distinct x1, x2 in unused such that:
            # 1. is_valid_addition(x1, mask_trial)
            # 2. is_valid_addition(x2, mask_trial | (1<<(x1-1)))
            
            # To speed up, pre-filter candidates for x1
            valid_x1 = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                # Try pairs
                # Shuffle to avoid bias
                random.shuffle(valid_x1)
                # Limit pairs checked to save time
                max_checks = 50
                checks = 0
                found = False
                for i in range(len(valid_x1)):
                    if checks > max_checks: break
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_x1)):
                        if checks > max_checks: break
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            # Success
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1 # -1 + 2 = +1 net gain
                            
                            # Update lists
                            current_elements.remove(x_out)
                            current_elements.append(x1)
                            current_elements.append(x2)
                            unused_elements.remove(x1)
                            unused_elements.remove(x2)
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                                
                            found = True
                            break
                    if found: break
                
                if found:
                    continue
        
        # If stuck, perturb: remove 2 random elements, add 1? Or just restart?
        # Let's try a random restart to a random valid set of similar size?
        # Or just perturb by removing a random element and hoping to add back.
        if current_elements:
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            current_mask = current_mask & ~(1 << (x_out - 1))
            current_size -= 1
            current_elements.remove(x_out)
            unused_elements.append(x_out)

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)
