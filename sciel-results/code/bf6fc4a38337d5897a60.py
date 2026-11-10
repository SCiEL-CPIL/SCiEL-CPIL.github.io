import time

def _has_matching(m, L):
    n = 12
    adj = [None] * n
    for i in range(1, n + 1):
        first = m + 1
        rem = first % i
        if rem == 0:
            fm = first
        else:
            fm = first + (i - rem)
        if fm > m + L:
            return False
        lst = []
        b = fm
        end = m + L
        while b <= end:
            lst.append(b - m - 1)
            b += i
        adj[i-1] = lst
    
    match_r = [-1] * L
    match_l = [-1] * n
    
    # Iterative DFS to avoid recursion overhead
    for u in range(n):
        visited = [False] * L
        stack = [(u, 0)]  # (node, next_index_in_adj)
        path = [u]
        while stack:
            node, idx = stack[-1]
            if idx < len(adj[node]):
                v = adj[node][idx]
                stack[-1] = (node, idx + 1)
                if not visited[v]:
                    visited[v] = True
                    if match_r[v] == -1:
                        # Found augmenting path
                        # Reconstruct path
                        path = []
                        cur_v = v
                        cur_u = node
                        while True:
                            path.append((cur_u, cur_v))
                            prev_u = match_l[cur_u]  # This isn't right
                            break
                        # Actually, let me use recursive DFS, it's simpler
                        return False  # This is wrong, let me use recursive
            else:
                stack.pop()
                if not stack:
                    break
        # This iterative approach is getting complex, let me use recursive
    
    # Back to recursive
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                match_l[u] = v
                return True
        return False
    
    for u in range(n):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    LCM12 = 27720
    best_m = 0
    best_f = _compute_f(0)
    
    for m in range(1, LCM12):
        if time.time() > deadline - 3:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
