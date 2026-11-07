import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 12
    # LCM(1..12) = 27720
    PERIOD = 27720
    MAX_L = 144
    
    def has_matching(m, L):
        # m is the start, interval is (m, m+L]
        # Left nodes: 0..11 corresponding to i=1..12
        # Right nodes: values in (m, m+L], indexed 0..L-1
        
        adj = [[] for _ in range(N)]
        
        for i in range(1, N + 1):
            first = ((m // i) + 1) * i
            if first > m + L:
                continue
            for v in range(first, m + L + 1, i):
                idx = v - (m + 1)
                adj[i - 1].append(idx)
                
        # Hopcroft-Karp algorithm
        match_to_left = [-1] * L  # match_to_right[j] = left node index
        match_to_right = [-1] * N  # match_to_left[i] = right node index
        dist = [0] * N
        
        from collections import deque
        
        def bfs():
            queue = deque()
            for i in range(N):
                if match_to_right[i] == -1:
                    dist[i] = 0
                    queue.append(i)
                else:
                    dist[i] = -1
            found = False
            while queue:
                i = queue.popleft()
                for j in adj[i]:
                    next_i = match_to_left[j]
                    if next_i == -1:
                        found = True
                    elif dist[next_i] == -1:
                        dist[next_i] = dist[i] + 1
                        queue.append(next_i)
            return found
        
        def dfs(i, seen):
            for j in adj[i]:
                next_i = match_to_left[j]
                if next_i == -1 or (dist[next_i] == dist[i] + 1 and dfs(next_i, seen)):
                    match_to_left[j] = i
                    match_to_right[i] = j
                    return True
            dist[i] = -1
            return False
        
        matching_count = 0
        while bfs():
            for i in range(N):
                if match_to_right[i] == -1:
                    seen = [False] * L
                    if dfs(i, seen):
                        matching_count += 1
        return matching_count == N

    best_L = 0
    best_m = 0
    
    for m in range(PERIOD):
        # Check time
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # L is at least 1. Max L is 144.
        for L in range(1, MAX_L + 1):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                break
        # If we found the maximum possible L, we can stop early
        if best_L == MAX_L:
            break
        
    return int(best_m)
