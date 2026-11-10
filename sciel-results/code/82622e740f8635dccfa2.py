import time

def construct():
    start_time = time.time()
    deadline = 150.0 - 10.0
    
    q = 5
    
    # Projective plane of order q: q^2 + q + 1 = 31 vertices
    vertices = []
    for y in range(q):
        for z in range(q):
            vertices.append((1, y, z))
    for z in range(q):
        vertices.append((0, 1, z))
    vertices.append((0, 0, 1))
    
    n = len(vertices)
    
    def bilinear(p, r):
        x1, y1, z1 = p
        x2, y2, z2 = r
        return (x1 * z2 + x2 * z1 + 2 * y1 * y2) % q
    
    # Build polarity edges
    polarity_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bilinear(vertices[i], vertices[j]) == 0:
                polarity_edges.append([i, j])
    
    target = 91
    num_pe = len(polarity_edges)
    
    if num_pe == target:
        return polarity_edges
    
    if num_pe > target:
        # Remove edges to reach target
        # Sort by how "central" they are (in more 4-cycles) and remove those first
        # For speed, just remove from the end
        return polarity_edges[:target]
    
    # num_pe < target, need to add edges
    existing = set(tuple(e) for e in polarity_edges)
    edges = list(polarity_edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate non-edges
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing:
                candidates.append((i, j))
    
    need = target - len(edges)
    
    for _ in range(need):
        if time.time() > deadline:
            break
        best_edge = None
        best_count = float('inf')
        for u, v in candidates:
            # Count new 4-cycles from adding (u,v)
            count = 0
            for x in adj[v]:
                if x == u:
                    continue
                for y in adj[u]:
                    if y == v or y == x:
                        continue
                    if y in adj[x]:
                        count += 1
            if count < best_count:
                best_count = count
                best_edge = (u, v)
        
        if best_edge is None:
            break
        u, v = best_edge
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        candidates.remove((u, v))
    
    return edges
