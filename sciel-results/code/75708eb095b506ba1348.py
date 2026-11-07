import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    n = 12
    period = 27720  # LCM(1..12)
    
    def check_matching(L, m):
        """Check if perfect matching exists in window (m, m+L]."""
        # Build bitmasks for adjacency
        adj_mask = [0] * n
        for i in range(1, n + 1):
            r = (-m - 1) % i
            mask = 0
            for j in range(r, L, i):
                mask |= (1 << j)
            adj_mask[i - 1] = mask
        
        matchR = [-1] * L
        
        def dfs(u, seen_mask):
            avail = adj_mask[u] & ~seen_mask
            while avail:
                lsb = avail & -avail
                v = lsb.bit_length() - 1
                avail ^= lsb
                seen_mask |= lsb
                if matchR[v] == -1 or dfs(matchR[v], seen_mask):
                    matchR[v] = u
                    return True
            return False
        
        # Process nodes with smaller degrees first (higher i)
        for u in range(n - 1, -1, -1):
            if not dfs(u, 0):
                return False
        return True
    
    def f_of_m(m):
        """Compute f(12, m): least positive L with perfect matching."""
        # Lower bound: max over i of distance to next multiple of i
        min_L = 0
        for i in range(1, n + 1):
            next_mult = ((m // i) + 1) * i
            dist = next_mult - m
            if dist > min_L:
                min_L = dist
        
        for L in range(max(min_L, 1), 145):  # L at most 144 per task
            if check_matching(L, m):
                return L
        return 144
    
    best_m = 0
    best_L = 0
    
    # Precompute all f(12, m) for m in [0, period-1]
    # Use the periodicity: f(12, m + period) = f(12, m)
    for m in range(period):
        if time.time() > deadline:
            break
        L_val = f_of_m(m)
        if L_val > best_L:
            best_L = L_val
            best_m = m
    
    return int(best_m)
