def construct():
    n = 15
    # We'll try to build a dense graph avoiding 3-edge paths with equal-degree endpoints
    # Strategy: start with complete graph, remove edges to fix violations
    
    import itertools
    import random
    
    def degrees_from_edges(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def has_violation(edges):
        deg = degrees_from_edges(edges)
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check all 3-edge paths s-u-v-t
        for u in range(n):
            for v in range(n):
                if u == v or not adj[u][v]:
                    continue
                for s in range(n):
                    if s == u or s == v or not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == u or t == v or t == s or not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            return True, (s, u, v, t)
        return False, None
    
    # Start with complete graph
    edges = [[i, j] for i in range(n) for j in range(i+1, n)]
    
    # Iteratively remove edges that cause violations
    changed = True
    while changed:
        changed = False
        violated, path = has_violation(edges)
        if violated:
            # Remove an edge from the violating path
            s, u, v, t = path
            # Try removing edge (u,v), (s,u), or (v,t)
            candidates = [(s,u), (u,v), (v,t)]
            removed = False
            for a, b in candidates:
                if [a,b] in edges or [b,a] in edges:
                    edges = [e for e in edges if not (e[0]==a and e[1]==b) and not (e[0]==b and e[1]==a)]
                    changed = True
                    removed = True
                    break
            if not removed:
                # Fallback: remove a random edge
                if edges:
                    edges.pop()
                    changed = True
    
    return edges
