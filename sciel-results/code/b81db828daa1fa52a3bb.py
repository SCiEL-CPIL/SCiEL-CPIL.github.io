def construct():
    mod = 1009
    # Compute inverses
    inv = [0] * (mod)
    for x in range(1, mod):
        inv[x] = pow(x, -1, mod)
        
    # min_M_for_m[m] = minimum M among all pairs with min(x, inv[x]) == m
    # m ranges from 1 to 1008
    INF = float('inf')
    min_M_for_m = [INF] * (mod)
    
    for x in range(1, mod):
        y = inv[x]
        m = min(x, y)
        M = max(x, y)
        if M < min_M_for_m[m]:
            min_M_for_m[m] = M
            
    # Compute suffix minimums
    # suffix_min[s] = min(min_M_for_m[s], min_M_for_m[s+1], ..., min_M_for_m[1008])
    suffix_min = [INF] * (mod + 1)
    suffix_min[mod] = INF
    for s in range(mod - 1, 0, -1):
        if min_M_for_m[s] < suffix_min[s + 1]:
            suffix_min[s] = min_M_for_m[s]
        else:
            suffix_min[s] = suffix_min[s + 1]
            
    best_start = 1
    best_len = 0
    
    for s in range(1, mod):
        B_s = suffix_min[s]
        if B_s == INF:
            e_max = mod - 1
        else:
            e_max = B_s - 1
            
        if e_max >= s:
            length = e_max - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    # If best_len is 0, it means no valid interval? 
    # But there are always valid intervals of length 1 unless the element is self-inverse.
    # Self-inverses are 1 and 1008.
    # If s=1, B_1 includes pair (1,1) so M=1, B_1=1, e_max=0, len=0.
    # If s=2, B_2 is min M for m>=2. 
    # The smallest M for m>=2 will be at least 2? 
    # Actually, if x=2, y=inv[2]. m=min(2,y). M=max(2,y).
    # Is it possible that all intervals have length 0? No, e.g., [2, 2] is valid if inv[2] != 2.
    # inv[2] = 505. m=2, M=505.
    # For s=2, B_2 <= 505. e_max >= 504. Length >= 503.
    
    return [best_start, best_len]
