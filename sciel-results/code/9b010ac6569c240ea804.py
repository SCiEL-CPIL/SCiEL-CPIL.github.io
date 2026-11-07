import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 12
    # LCM(1..12) = 27720
    PERIOD = 27720
    
    def has_matching(m, L):
        # m is the start, interval is (m, m+L]
        # Left nodes: 0..11 corresponding to i=1..12
        # Right nodes: values in (m, m+L]
        
        adj = [[] for _ in range(N)]
        
        for i in range(1, N + 1):
            first = ((m // i) + 1) * i
            if first > m + L:
                continue
            for v in range(first, m + L + 1, i):
                idx = v - (m + 1)
                adj[i - 1].append(idx)
                
        # Standard DFS Bipartite Matching
        match_to_left = [-1] * L  # match_to_right[j] = i
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_to_left[j] == -1 or dfs(match_to_left[j], seen):
                    match_to_left[j] = i
                    return True
            return False
        
        count = 0
        for i in range(N):
            seen = [False] * L
            if dfs(i, seen):
                count += 1
            else:
                return False
        return count == N

    best_L = 0
    best_m = 0
    
    for m in range(PERIOD):
        # Check time
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # L is at least 1. Max L is 144.
        for L in range(1, 145):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
        # If we found the maximum possible L, we can stop early
        if best_L == 144:
            break
        
    return int(best_m)
