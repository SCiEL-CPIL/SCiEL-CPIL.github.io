import math
import time
import bisect

def construct():
    start_time = time.time()
    # Reserve 15s for sorting and search, leave 45s for generation
    gen_deadline = start_time + 45
    
    # Compute LCM of 1 to 30
    D = 1
    for i in range(1, 31):
        D = D * i // math.gcd(D, i)
    
    # Compute weights w_i = D / i
    weights = [D // i for i in range(1, 31)]
    
    # Split weights into two halves
    W1 = weights[:15]
    W2 = weights[15:]
    
    N1 = 3 ** 15
    
    # Generate all sums for first half
    d = [0] * 15
    S1 = sum(-w for w in W1)
    list1 = [(S1, 0)]
    
    for k in range(1, N1):
        # Increment base-3 number d
        i = 0
        while i < 15 and d[i] == 2:
            d[i] = 0
            i += 1
        if i < 15:
            d[i] += 1
        
        delta = 0
        for j in range(i):
            delta -= 2 * W1[j]
        if i < 15:
            delta += W1[i]
        
        S1 += delta
        list1.append((S1, k))
        
        if time.time() > gen_deadline:
            break
    
    # Sort list1 by sum
    list1.sort(key=lambda x: x[0])
    sums1 = [x[0] for x in list1]
    
    # Generate all sums for second half
    d2 = [0] * 15
    S2 = sum(-w for w in W2)
    
    best_N = float('inf')
    best_coeffs = None
    
    search_deadline = start_time + 58  # Leave 2s margin
    
    for k in range(0, N1):
        target = -S2
        pos = bisect.bisect_left(sums1, target)
        
        candidates = set()
        for offset in [-2, -1, 0, 1, 2]:
            p = pos + offset
            if 0 <= p < len(sums1):
                candidates.add(p)
        
        for p in candidates:
            S1_val, idx1 = list1[p]
            N = S1_val + S2
            if N != 0 and abs(N) < best_N:
                best_N = abs(N)
                coeffs = [0] * 30
                tmp = idx1
                for j in range(15):
                    tmp, r = divmod(tmp, 3)
                    coeffs[j] = r - 1
                tmp = k
                for j in range(15):
                    tmp, r = divmod(tmp, 3)
                    coeffs[15 + j] = r - 1
                best_coeffs = coeffs
        
        # Increment base-3 number d2
        if k < N1 - 1:
            i = 0
            while i < 15 and d2[i] == 2:
                d2[i] = 0
                i += 1
            if i < 15:
                d2[i] += 1
            
            delta = 0
            for j in range(i):
                delta -= 2 * W2[j]
            if i < 15:
                delta += W2[i]
            S2 += delta
        
        if time.time() > search_deadline:
            break
    
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
    
    return best_coeffs
