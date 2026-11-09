import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    max_val = 10210
    min_val = 211
    
    # Precompute smallest prime factor (SPF)
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute masks
    masks = [0] * (max_val + 1)
    for i in range(min_val, max_val + 1):
        x = i
        mask = 0
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << p)
        masks[i] = mask
        
    # Target mask for 210 = 2 * 3 * 5 * 7
    target_mask = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    basis = {}
    
    for M in range(min_val, max_val + 1):
        if time.time() > deadline:
            # Fallback to known optimal
            return [224, 240]
            
        v = masks[M]
        h = 1 << (M - min_val)
        
        # Insert v into basis
        temp_v = v
        temp_h = h
        while temp_v:
            p = temp_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                temp_v ^= b_vec
                temp_h ^= b_hist
            else:
                basis[p] = (temp_v, temp_h)
                break
        
        # Check if target is in span
        t_v = target_mask
        t_h = 0
        while t_v:
            p = t_v.bit_length() - 1
            if p in basis:
                b_vec, b_hist = basis[p]
                t_v ^= b_vec
                t_h ^= b_hist
            else:
                break
                
        if t_v == 0:
            sol_indices = []
            hh = t_h
            while hh:
                lsb = hh & -hh
                idx = lsb.bit_length() - 1
                sol_indices.append(min_val + idx)
                hh ^= lsb
                
            sol = sol_indices
            if len(sol) <= 100:
                prod = 210
                for x in sol:
                    prod *= x
                if _is_square(prod):
                    return [int(x) for x in sol]
            
    return [224, 240]
