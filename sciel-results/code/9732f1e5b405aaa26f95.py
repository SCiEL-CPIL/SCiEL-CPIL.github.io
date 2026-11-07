import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    max_val = n
    
    # Precompute pairs_sum[s] = list of (a, b) such that a + b = s, a < b
    pairs_sum = [[] for _ in range(2 * max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= 2 * max_val:
                pairs_sum[s].append((a, b))
    
    # Precompute compat[x]: bitmask where bit (b-1) is set iff b + x <= max_val
    # This means: for value x, compat[x] has bit (b-1) set for all b such that x+b <= max_val
    # Then A_bits = mask & compat[x] gives us {b in S | x+b <= max_val}
    # But we also need x+b in S, so we need to check mask & (mask shifted)
    # Actually, let's precompute shift_mask[x]: bitmask where bit (x+b-1) is set for valid b
    # So shift_mask[x] has bits at positions (x+b-1) for b in 1..max_val-x
    # Then A_bits = mask & (shift_mask[x] >> x) ... no, that's not right either.
    
    # Better approach: For each x, precompute sum_mask[x] where bit (b-1) is set iff x+b <= max_val
    # Then A_bits = mask & sum_mask[x] & (mask shifted right by x)
    # Wait, let me think again.
    
    # A = {b in S | x+b in S}
    # b is in A iff: bit (b-1) in mask AND bit (x+b-1) in mask
    # Let's define: for each x, precompute mask_x where bit (b-1) is set iff x+b <= max_val
    # Then A_bits = mask & mask_x & (mask >> x) ... no.
    
    # Actually: bit (x+b-1) in mask means the value x+b is in S.
    # If we shift mask right by x, then bit (b-1) in (mask >> x) is the same as bit (x+b-1) in mask.
    # So A_bits = mask & (mask >> x) & valid_range_mask[x]
    # where valid_range_mask[x] has bits 0 to (max_val-x-1) set.
    
    # Let's precompute:
    # valid_range_mask[x]: bits 0..(max_val-x-1) set
    valid_range_mask = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        if x <= max_val:
            num_bits = max_val - x
            valid_range_mask[x] = (1 << num_bits) - 1
    
    def is_valid_addition(x, mask):
        """
        Check if adding x to valid set 'mask' results in a valid set.
        x is 1-indexed. mask is bitset where bit i corresponds to value i+1.
        """
        if mask & (1 << (x - 1)):
            return True  # Already in set
            
        x_bit = 1 << (x - 1)
        
        # Case 1: x is one of the members a,b,c.
        # WLOG a=x. Need b,c in S such that b+c in S, x+b in S, x+c in S.
        # A = {b in S | x+b in S}
        # A_bits = mask & (mask >> x) & valid_range_mask[x]
        
        A_bits = mask & (mask >> x) & valid_range_mask[x]
        
        # Now check if there exist distinct b, c in A_bits such that b+c in S
        # For each b in A_bits, check if there's c in A_bits, c != b, with b+c in S
        # b+c in S means bit (b+c-1) in mask is set.
        
        m = A_bits
        while m:
            lsb_b = m & -m
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m ^= lsb_b
            
            # Need c in A_bits, c != b, such that b+c in S
            # b+c ranges from b+1 to b+(max_val-b) = max_val
            # c must be in A_bits and c != b, and bit (b+c-1) in mask must be set
            
            # Let C_b = {c in S | b+c in S} = mask & (mask >> b) & valid_range_mask[b]
            # We need A_bits & C_b to have a bit other than b's bit.
            
            C_b = mask & (mask >> b_val) & valid_range_mask[b_val]
            common = A_bits & C_b
            if common & ~lsb_b:
                return False
        
        # Case 2: x is one of the sums.
        # WLOG a+b = x. a,b in S.
        # Need c in S, c != a, c != b, such that a+c in S and b+c in S.
        
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
                
            # U_a = {c in S | a+c in S}
            U_a = mask & (mask >> a) & valid_range_mask[a]
            U_b = mask & (mask >> b) & valid_range_mask[b]
            
            common = U_a & U_b
            a_bit = 1 << (a - 1)
            b_bit = 1 << (b - 1)
            if common & ~(a_bit | b_bit):
                return False
                
        return True

    def popcount(x):
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count

    # Initial valid set: Upper half
    best_mask = 0
    for i in range(n // 2, n):
        best_mask |= (1 << i)
    best_size = n - n // 2
    
    current_mask = best_mask
    current_size = best_size
    
    unused_elements = [i for i in range(1, n + 1) if not (current_mask & (1 << (i - 1)))]
    current_elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
    
    # Local search loop
    no_progress_count = 0
    max_no_progress = 50
    
    while time.time() < deadline:
        # Try greedy add: find any unused element that can be added
        if unused_elements:
            random.shuffle(unused_elements)
            added = False
            for x in unused_elements:
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    unused_elements.remove(x)
                    current_elements.append(x)
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    no_progress_count = 0
                    break
            if added:
                continue
        
        # Try 1-out, 2-in swap
        if current_elements and unused_elements:
            # Pick a random element to remove
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            # Find all unused elements that can be added to mask_trial
            valid_x1 = []
            for x in unused_elements:
                if is_valid_addition(x, mask_trial):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                random.shuffle(valid_x1)
                found = False
                max_checks = 200
                checks = 0
                for i in range(len(valid_x1)):
                    if checks > max_checks:
                        break
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, len(valid_x1)):
                        if checks > max_checks:
                            break
                        checks += 1
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            # Accept the swap
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            
                            current_elements.remove(x_out)
                            current_elements.append(x1)
                            current_elements.append(x2)
                            unused_elements.remove(x1)
                            unused_elements.remove(x2)
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            
                            found = True
                            no_progress_count = 0
                            break
                    if found:
                        break
                
                if found:
                    continue
        
        # Perturbation: remove a random element
        if current_elements:
            idx_out = random.randint(0, len(current_elements) - 1)
            x_out = current_elements[idx_out]
            current_mask &= ~(1 << (x_out - 1))
            current_size -= 1
            current_elements.remove(x_out)
            unused_elements.append(x_out)
            no_progress_count += 1
        
        if no_progress_count > max_no_progress:
            # Reset to best and restart local search
            current_mask = best_mask
            current_size = best_size
            current_elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
            unused_elements = [i for i in range(1, n + 1) if not (current_mask & (1 << (i - 1)))]
            no_progress_count = 0
    
    # Final systematic pass on best_mask
    remaining_time = deadline - time.time()
    if remaining_time > 3:
        final_mask = best_mask
        final_size = best_size
        final_unused = [i for i in range(1, n + 1) if not (final_mask & (1 << (i - 1)))]
        
        progress = True
        while progress and time.time() < deadline - 1:
            progress = False
            for x in list(final_unused):
                if is_valid_addition(x, final_mask):
                    final_mask |= (1 << (x - 1))
                    final_size += 1
                    final_unused.remove(x)
                    if final_size > best_size:
                        best_size = final_size
                        best_mask = final_mask
                    progress = True
        
        # Try 1-out, 2-in on best
        if time.time() < deadline - 1:
            for _ in range(30):
                if time.time() > deadline - 1:
                    break
                final_elements = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
                final_unused = [i for i in range(1, n + 1) if not (best_mask & (1 << (i - 1)))]
                
                if not final_elements or len(final_unused) < 2:
                    break
                    
                idx_out = random.randint(0, len(final_elements) - 1)
                x_out = final_elements[idx_out]
                mask_trial = best_mask & ~(1 << (x_out - 1))
                
                valid_x1 = []
                for x in final_unused:
                    if is_valid_addition(x, mask_trial):
                        valid_x1.append(x)
                
                if len(valid_x1) >= 2:
                    random.shuffle(valid_x1)
                    for i in range(len(valid_x1)):
                        x1 = valid_x1[i]
                        mask_1 = mask_trial | (1 << (x1 - 1))
                        for j in range(i + 1, len(valid_x1)):
                            x2 = valid_x1[j]
                            if is_valid_addition(x2, mask_1):
                                new_mask = mask_1 | (1 << (x2 - 1))
                                if popcount(new_mask) > best_size:
                                    best_size = popcount(new_mask)
                                    best_mask = new_mask
                                break
                        else:
                            continue
                        break

    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)
