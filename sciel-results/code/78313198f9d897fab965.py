import time

def construct():
    start_time = time.time()
    limit = 88.0  # Leave 2s margin
    
    N = 12
    period = 27720
    max_f = 144
    
    def has_matching(m, L):
        # Compute residues
        # r[i] is the offset for i+1
        r = [0] * N
        for i in range(N):
            r[i] = (-m - 1) % (i + 1)
            
        cands = [None] * N
        for i in range(N):
            ri = r[i]
            if ri >= L:
                return False
            cands[i] = list(range(ri, L, i + 1))
            
        # Order by fewest candidates
        order = list(range(N))
        order.sort(key=lambda i: len(cands[i]))
        
        match_to = [-1] * L
        
        # Iterative or recursive DFS? Recursive is fine for depth 12.
        def dfs(i, visited):
            for j in cands[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to[j] == -1 or dfs(match_to[j], visited):
                    match_to[j] = i
                    return True
            return False
            
        for i in order:
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True

    def compute_f(m):
        lo, hi = 12, max_f
        best = max_f
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(m, mid):
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return best

    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
            
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f == max_f:
                break

    return best_m
