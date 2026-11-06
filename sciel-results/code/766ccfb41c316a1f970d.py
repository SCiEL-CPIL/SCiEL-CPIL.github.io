import math
import time

def construct():
    start = time.time()
    DEADLINE = start + 50.0
    
    D = 1
    for d in range(2, 49):
        D = D * d // math.gcd(D, d)
        
    denoms = list(range(2, 49))
    w = [D // d for d in denoms]
    n = len(w)
    
    n1 = 23
    n2 = n - n1
    
    w1 = w[:n1]
    w2 = w[n1:]
    
    # Generate packed values for second half with pruning
    # Packed: (sum << 24) | mask
    MASK_BITS = 24
    MASK_LIMIT = (1 << MASK_BITS) - 1
    
    packed2 = [0]
    for i in range(n2):
        wi = w2[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) | mi
        # Prune: only keep values where sum < D
        # For each existing p, new value is p + delta
        # sum_new = (p + delta) >> MASK_BITS = (p >> MASK_BITS) + wi
        # So we can check if (p >> MASK_BITS) + wi < D
        new_packed2 = []
        for p in packed2:
            s = p >> MASK_BITS
            if s + wi < D:
                new_packed2.append(p + delta)
            new_packed2.append(p)
        packed2 = new_packed2
        
        if time.time() > DEADLINE - 15:
            break
    
    packed2.sort()
    
    # Generate first half
    packed1 = [0]
    for i in range(n1):
        wi = w1[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) | mi
        new_packed1 = []
        for p in packed1:
            s = p >> MASK_BITS
            if s + wi < D:
                new_packed1.append(p + delta)
            new_packed1.append(p)
        packed1 = new_packed1
        
        if time.time() > DEADLINE - 5:
            break
    
    packed1.sort()
    
    # Two-pointer search
    best_W = -1
    best_mask = 0
    
    i = 0
    j = len(packed2) - 1
    len1 = len(packed1)
    len2 = len(packed2)
    
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
                best_mask = m1 | (m2 << n1)
            i += 1
        else:
            j -= 1
            
    if best_mask == 0:
        return [0]
        
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i)
            
    return result
