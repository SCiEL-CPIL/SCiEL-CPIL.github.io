import time

def construct():
    start = time.time()
    deadline = 110.0

    # 8 disjoint K4s: vertices 0-31
    # K4_i: vertices {4*i, 4*i+1, 4*i+2, 4*i+3}
    # This gives n=32, alpha=8 (pick at most 1 from each K4), no C5 (K4 has no C5)
    
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([4 * b + i, 4 * b + j])
    
    n = 32
    
    # Verify no C5 and alpha < 9
    if time.time() - start > deadline:
        return {'n': n, 'edges': edges}
    
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Check no C5
    def has_c5(adj, n):
        for a in range(n):
            for b in range(a + 1, n):
                if not adj[a][b]:
                    continue
                for c in range(b + 1, n):
                    if not adj[b][c]:
                        continue
                    for d in range(c + 1, n):
                        if not adj[c][d]:
                            continue
                        for e in range(d + 1, n):
                            if not adj[d][e]:
                                continue
                            if adj[e][a]:
                                return True
        return False
    
    # Check independence number
    def max_independent(adj, n):
        best = [0]
        def dfs(idx, current, size):
            if size + (n - idx) <= best[0]:
                return
            if idx == n:
                best[0] = max(best[0], size)
                return
            # Option 1: don't include idx
            dfs(idx + 1, current, size)
            # Option 2: include idx if no conflict
            can_include = True
            for v in current:
                if adj[idx][v]:
                    can_include = False
                    break
            if can_include:
                dfs(idx + 1, current + [idx], size + 1)
        
        dfs(0, [], 0)
        return best[0]
    
    if has_c5(adj, n):
        return {'n': n, 'edges': edges}
    
    alpha = max_independent(adj, n)
    if alpha >= 9:
        return {'n': n, 'edges': edges}
    
    return {'n': n, 'edges': edges}
