import time

P = 27720  # LCM(1..12)

def has_matching(m, L):
    # Build adjacency lists
    # adj[i] contains j in [0, L) such that (m + 1 + j) % (i + 1) == 0
    adj = [[] for _ in range(12)]
    base = m + 1
    for i in range(12):
        div = i + 1
        # Optimization: calculate first valid j, then step by div
        # (base + j) % div == 0
        # j = k*div - base
        # We need 0 <= j < L
        # Let r = base % div
        # We need (r + j) % div == 0 => j % div == (-r) % div
        target = (-base) % div
        # First j >= 0
        # If target == 0, j=0 is valid.
        # Else j=target.
        # But we must check if target < L.
        # Then add target + div, etc.
        
        # This is faster than checking every j
        j = target
        while j < L:
            adj[i].append(j)
            j += div
            
    # Kuhn's Algorithm (DFS)
    # Left side: 12 nodes (indices 0..11)
    # Right side: L nodes (indices 0..L-1)
    match_to = [-1] * L
    match_from = [-1] * 12
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_to[j] == -1 or dfs(match_to[j], visited):
                match_to[j] = i
                match_from[i] = j
                return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def construct():
    start_time = time.time()
    # Leave 10 seconds margin for safety
    deadline = start_time + 50.0
    
    best_m = 0
    best_L = 0
    
    # We only need to check m in [0, P) because of periodicity
    for m in range(P):
        if time.time() > deadline:
            break
            
        # Binary search for min L
        # L must be at least 12 (pigeonhole) and at most 144
        lo, hi = 12, 144
        min_L = 145
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
                
        if min_L > best_L:
            best_L = min_L
            best_m = m
            if best_L == 144:
                break
                
    return best_m
