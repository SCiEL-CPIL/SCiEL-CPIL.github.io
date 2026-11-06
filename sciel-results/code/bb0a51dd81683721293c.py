import math
import time

def construct():
    start = time.time()
    DEADLINE = start + 55.0
    
    D = 1
    for d in range(2, 49):
        D = D * d // math.gcd(D, d)
        
    denoms = list(range(2, 49))
    w = [D // d for d in denoms]
    n = len(w)
    
    # Sort by weight descending to improve pruning
    indexed_w = sorted(enumerate(w), key=lambda x: -x[1])
    sorted_indices = [idx for idx, _ in indexed_w]
    sorted_w = [wt for _, wt in indexed_w]
    
    n1 = n // 2
    n2 = n - n1
    
    w1 = sorted_w[:n1]
    w2 = sorted_w[n1:]
    
    # Use 26 bits for mask to fit 2^26 subsets
    MASK_BITS = 26
    MASK_LIMIT = (1 << MASK_BITS) - 1
    
    # Generate packed values for second half with pruning
    packed2 = [0]
    for i in range(n2):
        wi = w2[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) | mi
        
        # Optimization: Calculate limit for pruning without shifting every element
        # We need s + wi < D  =>  s < D - wi
        # p = (s << MASK_BITS) | m
        # Condition: p < ((D - wi) << MASK_BITS) | MASK_LIMIT
        limit = ((D - wi) << MASK_BITS) | MASK_LIMIT
        
        new_packed2 = []
        append = new_packed2.append
        for p in packed2:
            if p < limit:
                append(p + delta)
            append(p)
        packed2 = new_packed2
        
        if time.time() > DEADLINE - 20:
            break
    
    packed2.sort()
    
    # Generate first half
    packed1 = [0]
    for i in range(n1):
        wi = w1[i]
        mi = 1 << i
        delta = (wi << MASK_BITS) | mi
        limit = ((D - wi) << MASK_BITS) | MASK_LIMIT
        
        new_packed1 = []
        append = new_packed1.append
        for p in packed1:
            if p < limit:
                append(p + delta)
            append(p)
        packed1 = new_packed1
        
        if time.time() > DEADLINE - 8:
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
    
    # Map back to original indices
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(sorted_indices[i])
            
    return result
