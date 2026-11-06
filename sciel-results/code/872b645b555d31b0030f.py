import math
import bisect
import time

def construct():
    start = time.time()
    DEADLINE = start + 55.0
    
    # Calculate D = lcm(2..48)
    D = 1
    for d in range(2, 49):
        D = D * d // math.gcd(D, d)
        
    denoms = list(range(2, 49))
    w = [D // d for d in denoms]
    n = len(w)
    
    # Split into two halves
    n1 = n // 2  # 23
    n2 = n - n1  # 24
    
    # Weights for first half
    w1 = w[:n1]
    # Weights for second half
    w2 = w[n1:]
    
    # Generate packed values for second half
    # Packed value: (sum << 24) | mask
    # Note: mask is 24 bits, so sum must be shifted by 24.
    # But sum can be large.
    MASK_BITS = 24
    MASK_LIMIT = (1 << MASK_BITS) - 1
    
    packed2 = [0]
    for i in range(n2):
        wi = w2[i]
        mi = 1 << i
        # Add weight and set mask bit
        # new_val = old_val + (wi << MASK_BITS) + mi
        # This works because mi bits are 0 in old_val's lower part for this i?
        # Yes, we iterate i from 0 to n2-1, so bit i is not set in previous masks.
        delta = (wi << MASK_BITS) + mi
        new_packed2 = []
        # Optimization: use list comprehension or loop
        # Loop is fine for 2^24
        for p in packed2:
            new_packed2.append(p)
            new_packed2.append(p + delta)
        packed2 = new_packed2
        
    if time.time() > DEADLINE:
        return [0]
        
    packed2.sort()
    
    # Generate packed values for first half
    packed1 = [0]
    for i in range(n1):
        wi = w1[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) + mi
        new_packed1 = []
        for p in packed1:
            new_packed1.append(p)
            new_packed1.append(p + delta)
        packed1 = new_packed1
        
        if time.time() > DEADLINE:
             break
             
    # Search
    # For each p1 in packed1:
    # s1 = p1 >> MASK_BITS
    # m1 = p1 & MASK_LIMIT
    # Limit_s2 = D - 1 - s1
    # Target_packed2 = Limit_s2 << MASK_BITS | MASK_LIMIT
    # idx = bisect.bisect_right(packed2, Target_packed2)
    # if idx > 0:
    #    p2_candidate = packed2[idx-1]
    #    s2 = p2_candidate >> MASK_BITS
    #    if s1 + s2 < D:
    #       if s1 + s2 > best_W:
    #          best_W = s1 + s2
    #          best_mask = m1 | ( (p2_candidate & MASK_LIMIT) << n1 )
    
    # Optimization: We can process packed1 in batches or just loop.
    # 2^24 iterations is ~16 million.
    # bisect is C-speed.
    # The loop itself is Python speed.
    
    # To speed up, we can sort packed1 and use two pointers?
    # If we sort packed1, we can't easily map back to mask? We have mask in packed1.
    # Two pointers:
    # Sort packed1.
    # Sort packed2 (already done).
    # We want max s1+s2 < D.
    # Let i be index in sorted packed1, j be index in sorted packed2.
    # Start i=0, j=len(packed2)-1.
    # While i < len(packed1) and j >= 0:
    #    s1 = packed1[i] >> MASK_BITS
    #    s2 = packed2[j] >> MASK_BITS
    #    if s1 + s2 < D:
    #        update best
    #        i += 1
    #    else:
    #        j -= 1
        
    # This is O(2^24) instead of O(2^24 * log 2^24).
    # Much faster!
    
    packed1.sort()
    
    i = 0
    j = len(packed2) - 1
    len1 = len(packed1)
    len2 = len(packed2)
    
    best_W = -1
    best_mask = 0
    
    while i < len1 and j >= 0:
        if time.time() > DEADLINE:
            break
            
        p1 = packed1[i]
        s1 = p1 >> MASK_BITS
        m1 = p1 & MASK_LIMIT
        
        p2 = packed2[j]
        s2 = p2 >> MASK_BITS
        m2 = p2 & MASK_LIMIT
        
        total_s = s1 + s2
        
        if total_s < D:
            if total_s > best_W:
                best_W = total_s
                # Combine masks
                # m1 is for first half (bits 0..n1-1)
                # m2 is for second half (bits 0..n2-1), needs to be shifted by n1
                best_mask = m1 | (m2 << n1)
            i += 1
        else:
            j -= 1
            
    if best_mask == 0:
        return [0]
        
    # Reconstruct indices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
