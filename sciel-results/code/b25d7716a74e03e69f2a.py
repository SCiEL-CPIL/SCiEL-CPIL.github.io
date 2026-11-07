import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    period = 27720
    best_L = 0
    best_m = 0
    
    # Precompute bit masks for each i (1..12) for each possible value mod i?
    # No, we build adj dynamically.
    
    for m in range(period):
        if time.time() > deadline:
            break
            
        # adj[i] is a bitmask of right nodes (0..143) connected to left node i
        # We only care about right nodes up to current L.
        # Since max L is 144, we can use 144-bit integers.
        
        adj = [0] * 12
        
        # Helper to add a right node j (0-indexed, value m+1+j) to adj
        # j corresponds to value v = m + 1 + j
        # We need to know which i divide v.
        
        # We will iterate L from 12 to 144.
        # But we can start from a lower bound?
        # Min L is at least max( (m%i + 1) if (m%i + 1) <= L else ... )?
        # Actually, for a specific i, the first multiple in (m, m+L] is at distance d_i.
        # d_i = (i - m % i) % i. If d_i == 0, d_i = i.
        # So L must be at least max(d_i).
        
        min_L = 0
        for i in range(1, 13):
            rem = m % i
            dist = i - rem if rem != 0 else i
            if dist > min_L:
                min_L = dist
        
        # If min_L > 144, then f(12, m) > 144? 
        # The problem states f(12, m) <= 144. 
        # Is it possible min_L > 144?
        # max dist is 11 (for i=12, m%12=11 -> dist=1). Wait.
        # dist is in 1..i. Max dist for i=12 is 12.
        # So min_L <= 12.
        # So we always start L at 12?
        # Wait, if m=0, i=12, dist=12.
        # If m=1, i=12, dist=11.
        # So min_L is always <= 12.
        
        found_L = -1
        
        # We need to populate adj for L = min_L.
        # But it's easier to just iterate L from 12 to 144 and update adj.
        # Since min_L <= 12, we start at 12.
        
        # Initialize adj for L=12
        # Right nodes j=0..11 correspond to values m+1 .. m+12
        for j in range(12):
            v = m + 1 + j
            # Determine which i divide v
            for i in range(1, 13):
                if v % i == 0:
                    adj[i-1] |= (1 << j)
        
        # Check matching for L=12
        if _has_matching_bitset(adj, 12):
            found_L = 12
        else:
            for L in range(13, 145):
                # Add new right node j = L-1, value v = m + L
                v = m + L
                j = L - 1
                for i in range(1, 13):
                    if v % i == 0:
                        adj[i-1] |= (1 << j)
                
                if _has_matching_bitset(adj, L):
                    found_L = L
                    break
        
        if found_L != -1:
            if found_L > best_L:
                best_L = found_L
                best_m = m
                if best_L == 144:
                    break
        
    return int(best_m)

def _has_matching_bitset(adj, L):
    # adj: list of 12 bitmasks
    # L: number of right nodes
    # match_to_right[j] = i if right node j is matched to left node i, else -1
    match_to_right = [-1] * L
    
    def dfs(i, visited):
        # i is left node index 0..11
        # visited is bitmask of right nodes visited
        avail = adj[i] & ~visited & ((1 << L) - 1)
        while avail:
            # Get lowest set bit
            v_bit = avail & -avail
            j = v_bit.bit_length() - 1
            
            visited |= v_bit
            
            if match_to_right[j] == -1 or dfs(match_to_right[j], visited):
                match_to_right[j] = i
                return True
            
            avail &= avail - 1
        
        return False
    
    for i in range(12):
        if not dfs(i, 0):
            return False
    return True
