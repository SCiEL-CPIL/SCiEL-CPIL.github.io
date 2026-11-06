import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    denoms = list(range(2, 49))
    N = len(denoms)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Split denominators into two halves
    mid = N // 2  # 23
    D1 = denoms[:mid]
    D2 = denoms[mid:]
    
    # Weights
    W1 = [L // d for d in D1]
    W2 = [L // d for d in D2]
    
    # Generate subset sums for D1
    # Pack as (sum << bits) | mask
    bits1 = len(D1)
    shift1 = bits1
    mask1_limit = (1 << bits1) - 1
    
    vals1 = [0]
    for i, w in enumerate(W1):
        # For each existing sum, create a new one with bit i set
        # We need to iterate over a copy or handle in-place carefully
        # Using list comprehension is faster
        new_vals = []
        for v in vals1:
            # v is packed
            # We just add w to the sum part
            # The mask part: we need to set bit i
            # But we are generating from scratch.
            # It is easier to generate (sum, mask) pairs first, then pack.
            pass
        
    # Let's use a different generation method
    # Generate list of (sum, mask)
    # This uses more memory but is clearer
    # Actually, packing during generation is tricky because we need to know the mask.
    
    # Better:
    # Generate sums and masks separately? No.
    
    # Let's stick to generating packed integers.
    # Start with [(0, 0)]
    # For each weight w at index i:
    #   new_list = []
    #   for (s, m) in current_list:
    #       new_list.append((s, m))
    #       new_list.append((s + w, m | (1 << i)))
    #   current_list = new_list
    
    # To optimize, we pack.
    # val = (s << bits) | m
    
    def gen_packed(weights, bits):
        vals = [0]  # (0 << bits) | 0
        for i, w in enumerate(weights):
            # We need to add w to the sum part of each existing val
            # and set the i-th bit in the mask part
            # val_new = (val + (w << bits)) | (1 << i)
            # Wait, val = (s << bits) | m
            # s_new = s + w
            # m_new = m | (1 << i)
            # val_new = ((s + w) << bits) | (m | (1 << i))
            # val_new = (s << bits) + (w << bits) | m | (1 << i)
            # val_new = val + (w << bits) | (1 << i)
            # This works because the mask bits don't overlap with the sum bits.
            
            new_vals = []
            for v in vals:
                new_vals.append(v)
                new_vals.append(v + (w << bits) | (1 << i))
            vals = new_vals
            if time.time() > deadline:
                break
        return vals

    vals1 = gen_packed(W1, bits1)
    vals1.sort()
    
    if time.time() > deadline:
        # Fallback
        return [0]
        
    bits2 = len(D2)
    shift2 = bits2
    mask2_limit = (1 << bits2) - 1
    
    vals2 = gen_packed(W2, bits2)
    vals2.sort()
    
    if time.time() > deadline:
        # Fallback
        return [0]

    # Two pointer search
    # vals1 sorted ascending
    # vals2 sorted ascending
    # We want max s1 + s2 < L
    
    i = len(vals1) - 1
    j = 0
    best_sum = 0
    best_v1 = 0
    best_v2 = 0
    
    # We iterate i from high to low?
    # Standard:
    # i = len(V)-1, j = 0
    # while i >= 0 and j < len(W):
    #   s = V[i] + W[j]
    #   if s < L:
    #     best = max(best, s)
    #     j += 1
    #   else:
    #     i -= 1
    
    # Here V and W are packed.
    # We need to extract sums.
    # s1 = v1 >> bits1
    # s2 = v2 >> bits2
    
    # To speed up, we can pre-extract sums?
    # Or just extract on the fly.
    
    while i >= 0 and j < len(vals2):
        v1 = vals1[i]
        v2 = vals2[j]
        
        s1 = v1 >> bits1
        s2 = v2 >> bits2
        
        total = s1 + s2
        
        if total < L:
            if total > best_sum:
                best_sum = total
                best_v1 = v1
                best_v2 = v2
            j += 1
        else:
            i -= 1
            
        if time.time() > deadline:
            break
            
    if best_sum == 0:
        # Should not happen, but fallback
        return [0]
        
    # Reconstruct indices
    m1 = best_v1 & mask1_limit
    m2 = best_v2 & mask2_limit
    
    indices = []
    for k in range(len(D1)):
        if m1 & (1 << k):
            indices.append(k)
            
    for k in range(len(D2)):
        if m2 & (1 << k):
            indices.append(mid + k)
            
    return indices
