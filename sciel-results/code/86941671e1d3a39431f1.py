import time
import math

def construct():
    start_time = time.time()
    limit = 80  # Leave 10s for safety
    
    N = 12
    period = 27720
    max_L = 144

    def compute_f(m):
        # Precompute masks for L = max_L
        # mask[i] has bit j set if x = j+1 is valid for i in 1..max_L
        # x % i == (-m) % i
        full_masks = [0] * (N + 1)
        
        for i in range(1, N + 1):
            ri = (-m) % i
            mask = 0
            if ri == 0:
                start_x = i
            else:
                start_x = ri
            
            if start_x <= max_L:
                for x in range(start_x, max_L + 1, i):
                    mask |= (1 << (x - 1))
            
            full_masks[i] = mask
        
        # current_L_mask is updated in the loop
        current_L_mask = 0

        def backtrack(used_pos_mask, remaining_idx_mask):
            if remaining_idx_mask == 0:
                return True
            
            # Find index with minimum remaining values
            best_idx = -1
            best_avail = 0
            best_count = 100
            
            r_mask = remaining_idx_mask
            while r_mask:
                lsb_idx = r_mask & (-r_mask)
                i_bit = lsb_idx.bit_length() - 1
                r_mask ^= lsb_idx
                
                # i_bit is 0..11, corresponding to i = 1..12
                avail = full_masks[i_bit + 1] & current_L_mask & (~used_pos_mask)
                count = avail.bit_count()
                
                if count == 0:
                    return False
                
                if count < best_count:
                    best_count = count
                    best_idx = i_bit
                    best_avail = avail
            
            # Try all options for best_idx
            avail = best_avail
            while avail:
                lsb_pos = avail & (-avail)
                avail ^= lsb_pos
                
                new_remaining = remaining_idx_mask ^ (1 << best_idx)
                new_used_pos = used_pos_mask | lsb_pos
                
                if backtrack(new_used_pos, new_remaining):
                    return True
            
            return False

        # Iterate L
        for L in range(1, max_L + 1):
            current_L_mask = (1 << L) - 1
            
            # Quick check: does every i have at least one option?
            feasible = True
            for i in range(1, N + 1):
                if (full_masks[i] & current_L_mask).bit_count() == 0:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            if backtrack(0, (1 << N) - 1):
                return L
        
        return max_L

    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return best_m
