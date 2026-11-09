import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = 110.0
    
    # 8 disjoint K4s: vertices 0-31
    # K4_i: vertices {4*i, 4*i+1, 4*i+2, 4*i+3}
    # Add vertex 32
    # Connect 32 to all 4 vertices of K4_0, K4_1, K4_2, K4_3
    # Connect 32 to vertex 4*4 = 16 (one vertex) of K4_4, K4_5, K4_6, K4_7
    
    edges = []
    # K4 cliques
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([4 * b + i, 4 * b + j])
    
    # Vertex 32 connections
    # All 4 vertices of K4_0..K4_3
    for b in range(4):
        for i in range(4):
            edges.append([32, 4 * b + i])
    # One vertex from K4_4..K4_7
    for b in range(4, 8):
        edges.append([32, 4 * b])
    
    n = 33
    
    # Verify no C5 and alpha < 9
    if time.time() - start > deadline:
        # Fallback to 8xK4
        return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
    
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
        # Use branch and bound
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
        # Fallback
        return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
    
    alpha = max_independent(adj, n)
    if alpha >= 9:
        # Fallback
        return {'n': 32, 'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]}
    
    return {'n': n, 'edges': edges}
