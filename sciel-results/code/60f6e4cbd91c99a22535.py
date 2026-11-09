def construct():
    p = 1009
    # Precompute inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    # MinB[s] will store the minimum value of inv[a] for all a >= s such that a < inv[a]
    # If no such a exists, it stores p (which acts as infinity)
    MinB = [p] * (p + 1)
    
    # Iterate backwards to compute suffix minimums
    for a in range(p - 1, 0, -1):
        current_min = MinB[a + 1]
        if a < inv[a]:
            if inv[a] < current_min:
                current_min = inv[a]
        MinB[a] = current_min
        
    best_len = 0
    best_start = 1
    
    # s must be at least 2 because 1 is a self-inverse
    # e must be at most 1007 because 1008 is a self-inverse
    for s in range(2, p):
        # The maximum e is determined by the smallest partner b of any a >= s
        # If MinB[s] is p, it means no such partner exists in the range, so we can go up to p-1
        # But we must also respect the upper bound 1007
        max_e = MinB[s] - 1
        if max_e > 1007:
            max_e = 1007
            
        if max_e >= s:
            current_len = max_e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
