def construct():
    mod = 1009
    inv = [0] * 1009
    for x in range(1, 1009):
        for y in range(1, 1009):
            if (x * y) % mod == 1:
                inv[x] = y
                break
    
    # Precompute L_x and R_x
    # We need min_R for each L value.
    # L_x can be from 1 to 1008.
    # Actually, we only care about L_x >= s.
    # So we can create a list of pairs (L_x, R_x).
    pairs = []
    for x in range(1, 1009):
        y = inv[x]
        l = min(x, y)
        r = max(x, y)
        pairs.append((l, r))
    
    # Sort pairs by L descending? Or just bucket by L.
    # Since L is in [1, 1008], we can use an array min_R_for_L of size 1009.
    # Initialize with infinity.
    INF = 1009
    min_R_for_L = [INF] * 1009
    
    for l, r in pairs:
        if r < min_R_for_L[l]:
            min_R_for_L[l] = r
            
    best_start = 1
    best_len = 0 # Problem says nonempty, so length >= 1. But let's start 0 to be safe, will update.
    
    # We want M(s) = min { R_x | L_x >= s }
    # We can compute this by iterating s from 1008 down to 1.
    current_min_R = INF
    
    for s in range(1008, 0, -1):
        # Update current_min_R with min_R_for_L[s]
        if min_R_for_L[s] < current_min_R:
            current_min_R = min_R_for_L[s]
        
        # M(s) is current_min_R
        M_s = current_min_R
        e = M_s - 1
        length = e - s + 1
        
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
