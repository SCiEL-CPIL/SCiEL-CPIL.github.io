import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # Identify pairs {u, v} with u < v and u != v
    pairs = []
    seen = set()
    for i in range(1, N + 1):
        if inv[i] == i:
            continue
        j = inv[i]
        if i < j:
            pairs.append((i, j))
        elif j < i and j not in seen:
            # We just want unique pairs.
            # Since inv is a permutation, if i<j we add (i,j).
            # If j<i, the pair (j,i) would have been added when we processed j (if j<i).
            # So we only add if i < j.
            pass
            
    # Correct pair generation
    pairs = []
    for i in range(1, N + 1):
        if inv[i] == i:
            continue
        j = inv[i]
        if i < j:
            pairs.append((i, j))
            
    best_len = 0
    best_start = 2
    
    # We know s=1 and e=1008 are invalid due to self-inverses.
    # So s in [2, 1007] and e in [2, 1007]
    
    for s in range(2, N): # s up to 1007
        if time.time() > deadline:
            break
            
        e_limit = 1007
        
        for u, v in pairs:
            if u >= s:
                # If u is in the interval, v cannot be.
                # So e must be < v.
                if v - 1 < e_limit:
                    e_limit = v - 1
                    
        if e_limit >= s:
            current_len = e_limit - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    if best_len == 0:
        # Fallback, though there should always be a valid interval of length 1
        # e.g., [2,2] is valid if 2*2 != 1 mod 1009. 4 != 1.
        best_start = 2
        best_len = 1

    return [best_start, best_len]
