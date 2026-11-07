def construct():
    n = 15
    # Start with complete graph
    edges = [[i, j] for i in range(n) for j in range(i+1, n)]
    
    def get_edge_set(edges):
        return set((min(u,v), max(u,v)) for u,v in edges)

    def compute_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def find_violations(edge_set):
        deg = compute_degrees(edge_set)
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
            
        violations = []
        # Path s-u-v-t
        # s != u, u != v, v != t
        # s != t
        # adj[s][u], adj[u][v], adj[v][t] must be True
        # deg[s] == deg[t]
        
        # Iterate over middle edge (u,v)
        for u in range(n):
            for v in range(n):
                if u >= v or not adj[u][v]:
                    continue
                # u < v to avoid duplicate middle edges, but paths are directed in terms of endpoints?
                # The path is s-u-v-t. The edge is {u,v}.
                # s is neighbor of u, t is neighbor of v.
                # s != v, t != u, s != t.
                for s in range(n):
                    if s == u or s == v or not adj[s][u]:
                        continue
                    for t in range(n):
                        if t == u or t == v or t == s or not adj[v][t]:
                            continue
                        if deg[s] == deg[t]:
                            # Violation found
                            # Edges involved: (s,u), (u,v), (v,t)
                            violations.append(((s,u), (u,v), (v,t)))
        return violations

    changed = True
    while changed:
        changed = False
        edge_set = get_edge_set(edges)
        violations = find_violations(edge_set)
        if not violations:
            break
        
        # Count edge frequencies in violations
        from collections import Counter
        edge_counts = Counter()
        for path in violations:
            for e in path:
                # Normalize edge
                a, b = e
                if a > b:
                    a, b = b, a
                edge_counts[(a,b)] += 1
        
        if not edge_counts:
            break
            
        # Find max count
        max_count = max(edge_counts.values())
        # Find all edges with max count
        candidates = [e for e, c in edge_counts.items() if c == max_count]
        
        # Pick one (e.g., first one)
        edge_to_remove = candidates[0]
        
        # Remove it
        a, b = edge_to_remove
        edges = [e for e in edges if not ((e[0]==a and e[1]==b) or (e[0]==b and e[1]==a))]
        changed = True
        
    return edges
