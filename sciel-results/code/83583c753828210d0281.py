import time

def _has_c5(n, adj):
    """Exact check for any 5-cycle as a subgraph."""
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False

def _has_independent_set(n, adj, k):
    """Exact decision: does an independent set of size k exist?"""
    if k <= 1:
        return True
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                adj_list[i].add(j)
                adj_list[j].add(i)
                
    # Sort vertices by degree to improve pruning
    # Lower degree vertices are more likely to be in independent sets
    order = sorted(range(n), key=lambda x: len(adj_list[x]))
    
    best_found = 0

    def dfs(cand, size):
        nonlocal best_found
        if size + len(cand) < k:
            return False
        if size >= best_found:
            best_found = size
        if not cand:
            return size >= k
            
        v = cand[0]
        rest = cand[1:]
        
        # Include v
        new_cand = [x for x in rest if x not in adj_list[v]]
        if dfs(new_cand, size + 1):
            return True
            
        # Exclude v
        if dfs(rest, size):
            return True
            
        return False

    return dfs(order, 0)

def _make_adj(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    return adj

def construct():
    start = time.time()
    deadline = start + 110.0

    # Construct 8 disjoint K4s
    blocks = 8
    k = 4
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
                
    adj = _make_adj(n, edges)

    # Verify
    if _has_c5(n, adj):
        # Fallback (should not happen)
        blocks = 7
        k = 4
        n = blocks * k
        edges = []
        for b in range(blocks):
            base = b * k
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append([base + i, base + j])
        adj = _make_adj(n, edges)

    if _has_independent_set(n, adj, 9):
        # Fallback
        blocks = 7
        k = 4
        n = blocks * k
        edges = []
        for b in range(blocks):
            base = b * k
            for i in range(k):
                for j in range(i + 1, k):
                    edges.append([base + i, base + j])
        adj = _make_adj(n, edges)

    if time.time() > deadline:
        return {'n': n, 'edges': edges}

    return {'n': n, 'edges': edges}
