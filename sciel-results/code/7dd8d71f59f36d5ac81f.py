import time

P = 27720  # LCM(1..12)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_m = 0
    best_L = 0
    
    # Precompute for each m in [0, P) and each possible L (12..144)
    # whether a perfect matching exists.
    # For efficiency, we use bitmask-based bipartite matching.
    
    # For a given m and L, adj[i] is a bitmask of positions j in [0,L)
    # where (m+1+j) % (i+1) == 0
    
    # We'll iterate m and for each m do binary search on L.
    # To speed up matching, use Hopcroft-Karp or bitmask DFS.
    
    # Bitmask approach for 12 left nodes:
    # match[i] = bitmask of which right nodes are matched to left node i (at most 1 bit)
    # Actually, for perfect matching in bipartite graph with 12 left, L right:
    # We can use DFS with bitmasks.
    
    def has_matching_bitmask(m, L):
        """Check if perfect matching exists using bitmask DFS."""
        # Build adjacency as bitmasks
        adj = [0] * 12
        base = m + 1
        for i in range(12):
            div = i + 1
            target = (-base) % div
            mask = 0
            j = target
            while j < L:
                mask |= (1 << j)
                j += div
            adj[i] = mask
        
        # DFS with backtracking using bitmasks
        # match_to[j] = which left node is matched to right node j, or -1
        # We use recursive DFS with bitmask for visited right nodes
        
        import sys
        sys.setrecursionlimit(10000)
        
        match_to = [-1] * L  # match_to[j] = i if right j matched to left i
        
        def dfs(i, visited_mask):
            # Try all right nodes j in adj[i] that haven't been visited
            candidates = adj[i] & ~visited_mask
            while candidates:
                # Get lowest set bit
                low_bit = candidates & (-candidates)
                j = low_bit.bit_length() - 1
                candidates ^= low_bit
                new_visited = visited_mask | low_bit
                if match_to[j] == -1:
                    match_to[j] = i
                    return True
                if dfs(match_to[j], new_visited):
                    match_to[j] = i
                    return True
            return False
        
        for i in range(12):
            match_to = [-1] * L
            if not dfs(i, 0):
                return False
        return True
    
    # Iterate over all m in the period
    for m in range(P):
        if time.time() > deadline:
            break
        
        # Binary search for minimum L
        lo, hi = 12, 144
        min_L = 145
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching_bitmask(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if min_L > best_L:
            best_L = min_L
            best_m = m
            if best_L == 144:
                break
    
    return int(best_m)
