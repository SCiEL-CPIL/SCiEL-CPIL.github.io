import time

def construct():
    n = 15
    deadline = time.time() + 85  # Safety margin
    
    # Start with K_{7,8}
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    def get_degree(e_set):
        deg = [0] * n
        for (a, b) in e_set:
            deg[a] += 1
            deg[b] += 1
        return deg
    
    def is_valid(e_set):
        deg = get_degree(e_set)
        adj = [set() for _ in range(n)]
        for (a, b) in e_set:
            adj[a].add(b)
            adj[b].add(a)
        # Check all paths s-u-v-t
        for (u, v) in e_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    if not is_valid(edges):
        # Should not happen, but safety check
        pass
    
    # Try to add edges greedily
    all_possible = set()
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges and (j, i) not in edges:
                all_possible.add((i, j))
    
    # Sort by some heuristic: prefer edges between high-degree vertices
    deg = get_degree(edges)
    # Try adding edges, check validity
    candidates = sorted(all_possible, key=lambda e: -(deg[e[0]] + deg[e[1]]))
    
    for e in candidates:
        if time.time() > deadline:
            break
        new_edges = edges | {e}
        if is_valid(new_edges):
            edges = new_edges
            deg = get_degree(edges)
            # Re-sort remaining candidates
            candidates = [c for c in candidates if c not in edges]
            candidates.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
    
    # Convert to list of lists
    result = []
    for (u, v) in edges:
        result.append([int(u), int(v)])
    
    return result
