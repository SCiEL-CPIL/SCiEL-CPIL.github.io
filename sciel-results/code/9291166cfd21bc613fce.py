import time

def construct():
    deadline = time.time() + 85  # Safety margin
    
    p = 1009
    N = p - 1  # 1008
    
    # 1. Precompute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # 2. Identify self-inverses and inverse pairs
    self_inverse = set()
    pairs = []
    
    for x in range(1, p):
        if (x * x) % p == 1:
            self_inverse.add(x)
        else:
            y = inv[x]
            if x < y:
                pairs.append((x, y))
    
    # 3. Precompute minimum blocking end for each start
    # min_b_from[s] = min { b : (a, b) in pairs, a >= s }
    # min_self_from[s] = min { x : x in self_inverse, x >= s }
    
    # Initialize with a large value (p is sufficient as max index is p-1)
    INF = p + 1
    min_b_at = [INF] * (p + 2)
    min_self_at = [INF] * (p + 2)
    
    for a, b in pairs:
        if b < min_b_at[a]:
            min_b_at[a] = b
            
    for x in self_inverse:
        min_self_at[x] = x
        
    min_b_from = [INF] * (p + 2)
    min_self_from = [INF] * (p + 2)
    
    # Scan from right to left to find minimums for suffixes
    for s in range(p, 0, -1):
        min_b_from[s] = min(min_b_at[s], min_b_from[s + 1])
        min_self_from[s] = min(min_self_at[s], min_self_from[s + 1])
        
    # 4. Find the longest valid interval
    best_length = 0
    best_start = 1
    
    for s in range(1, p):
        if time.time() > deadline:
            break
            
        # If s is a self-inverse, any interval starting at s containing s is invalid.
        # Since the interval must be non-empty and start at s, it contains s.
        # So length is 0.
        if s in self_inverse:
            continue
        
        # The interval [s, e] is valid if e < min_b_from[s] and e < min_self_from[s]
        # So max e is min(min_b_from[s], min_self_from[s]) - 1
        max_e = min(min_b_from[s], min_self_from[s]) - 1
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
    # Fallback (should not happen as there are always valid intervals of length 1)
    if best_length == 0:
        for x in range(1, p):
            if x not in self_inverse:
                return [x, 1]
        return [2, 1]
        
    return [best_start, best_length]
