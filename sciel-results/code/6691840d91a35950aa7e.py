import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Sieve smallest prime factor (SPF)
    max_val = 10210
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Compute masks for all numbers 211..10210
    # We need to map prime to bit index.
    # Since we use SPF, we can just use the prime value as the bit index?
    # No, max prime is 10210. 1 << 10210 is huge but Python handles it.
    # However, using a dictionary to map prime to a dense index is better for bit operations?
    # Actually, 1 << 10210 is a 10KB integer. XORing 10KB integers is fine.
    # Let's just use the prime value as the bit position.
    # mask = sum(1 << p for p in square_free_part)
    
    masks = [0] * (max_val + 1)
    for x in range(2, max_val + 1):
        n = x
        m = 0
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                m |= (1 << p)
        masks[x] = m
        
    # 3. Target mask for 210
    # 210 = 2 * 3 * 5 * 7
    target_mask = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)
    
    # 4. Incremental Gaussian Elimination
    # basis: list of masks
    # combos: list of sets of indices (relative to 211)
    # pivot_map: {bit_position: index_in_basis}
    
    basis = []
    combos = []
    pivot_map = {}
    
    # We iterate M from 211 to 10210
    # index i corresponds to number 211 + i
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
            
        i = M - 211
        m = masks[M]
        c = {i}
        
        # Reduce m using existing basis
        while m != 0:
            hb = m.bit_length() - 1
            if hb in pivot_map:
                idx = pivot_map[hb]
                m ^= basis[idx]
                c ^= combos[idx]
            else:
                break
        
        if m != 0:
            hb = m.bit_length() - 1
            new_idx = len(basis)
            basis.append(m)
            combos.append(c)
            pivot_map[hb] = new_idx
            
        # Check if target is in span
        t = target_mask
        tc = set()
        while t != 0:
            hb = t.bit_length() - 1
            if hb in pivot_map:
                idx = pivot_map[hb]
                t ^= basis[idx]
                tc ^= combos[idx]
            else:
                break
        
        if t == 0:
            # Solution found
            # tc contains indices relative to 211
            # Convert to numbers
            selected_nums = [idx + 211 for idx in tc]
            
            # Verify
            # Check distinct and range
            if len(set(selected_nums)) == len(selected_nums):
                if all(210 < n <= 10210 for n in selected_nums):
                    prod = 1
                    for n in selected_nums:
                        prod *= n
                    val = 210 * prod
                    s = math.isqrt(val)
                    if s * s == val:
                        return [int(n) for n in selected_nums]
                        
    # Fallback
    return [840]
