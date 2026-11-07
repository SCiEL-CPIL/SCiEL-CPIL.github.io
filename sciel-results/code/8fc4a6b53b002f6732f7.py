import time

def construct():
    N = 12
    PERIOD = 27720
    MAX_L = 144
    
    def has_matching(m, L):
        candidates = [None] * N
        
        for i in range(1, N + 1):
            rem = m % i
            if rem == 0:
                start = m + i
            else:
                start = m + (i - rem)
            
            if start > m + L:
                return False
            
            candidates[i-1] = list(range(start, m + L + 1, i))
        
        match_to = [None] * N
        match_from = {}
        
        def dfs(i, visited):
            for x in candidates[i]:
                if visited.get(x, False):
                    continue
                visited[x] = True
                if x not in match_from or dfs(match_from[x], visited):
                    match_to[i] = x
                    match_from[x] = i
                    return True
            return False
        
        for i in range(N):
            visited = {}
            if not dfs(i, visited):
                return False
        return True

    best_m = 0
    best_L = 0
    start_time = time.time()
    
    for m in range(PERIOD):
        if time.time() - start_time > 45:
            break
            
        # Optimization: if we already found MAX_L, we can't improve
        if best_L == MAX_L:
            break
            
        lo, hi = 1, MAX_L
        current_max_L = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                current_max_L = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        if current_max_L > best_L:
            best_L = current_max_L
            best_m = m
            
    return int(best_m)
