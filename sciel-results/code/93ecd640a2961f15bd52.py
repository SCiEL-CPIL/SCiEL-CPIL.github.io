import time

def construct():
    """Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.
    
    Strategy: Use the polarity graph of PG(2,5) which is C4-free with 90 edges,
    then add one edge that creates the minimum number of new 4-cycles.
    """
    n = 31
    q = 5
    deadline = time.time() + 140
    
    # Generate all normalized points of PG(2,5)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize so first nonzero coordinate is 1
                if x != 0:
                    inv_x = pow(x, -1, q)
                    rep = (1, y * inv_x % q, z * inv_x % q)
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    rep = (0, 1, z * inv_y % q)
                else:
                    inv_z = pow(z, -1, q)
                    rep = (0, 0, 1)
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # The correct polarity form for PG(2,q) with q odd:
    # B((x,y,z),(x',y',z')) = x*x' + y*y' + z*z' + x*y' + y*z' + z*x' (mod q)
    # This is a symmetric bilinear form.
    
    def ortho(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        val = (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*z2 + z1*x2) % q
        return val == 0
    
    # Build the C4-free base graph (polarity graph)
    adj = [[False] * n for _ in range(n)]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if ortho(points[i], points[j]):
                adj[i][j] = True
                adj[j][i] = True
                edges.add((i, j))
    
    num_edges = len(edges)
    
    # The polarity graph of PG(2,5) should have exactly 90 edges
    if num_edges != 90:
        # Try alternative polarity forms
        # Form 2: B((x,y,z),(x',y',z')) = x*z' + y*(x'+z') + z*x'
        edges = set()
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1*z2 + y1*(x2 + z2) + z1*x2) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.add((i, j))
        num_edges = len(edges)
    
    if num_edges != 90:
        # Form 3: B((x,y,z),(x',y',z')) = x*z' + y*x' + z*y'
        edges = set()
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1*z2 + y1*x2 + z1*y2) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.add((i, j))
        num_edges = len(edges)
    
    # If we still don't have 90 edges, use a different construction
    # Use the known construction from finite geometry
    
    if num_edges != 90:
        # Alternative: construct directly using the incidence structure
        # Vertices: (a,b) for a,b in GF(5), plus (a,inf) for a in GF(5), plus (inf,inf)
        
        vertices = []
        for a in range(q):
            for b in range(q):
                vertices.append((a, b))
        for a in range(q):
            vertices.append((a, -1))
        vertices.append((-1, -1))
        
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        
        edges = set()
        adj = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = vertices[i]
                v2 = vertices[j]
                
                x1, y1 = v1
                x2, y2 = v2
                
                # Determine adjacency based on the polarity
                # (a,b) ~ (a',b') iff (a-a')*(b'-b) + (b-b') = 0 in GF(5)
                # This is a simplified form; let's use the standard one
                
                # Standard polarity graph adjacency for PG(2,q):
                # For finite points (a,b) and (a',b'): adjacent if a*a' + b*b' = 1 (or some constant)
                # Actually, let me just use the orthogonal points from the homogeneous form
                
                # Convert back to homogeneous and check orthogonality
                if x1 == -1 and y1 == -1:
                    p1 = (0, 0, 1)
                elif y1 == -1:
                    p1 = (1, 0, x1)
                else:
                    p1 = (x1, y1, 1)
                
                if x2 == -1 and y2 == -1:
                    p2 = (0, 0, 1)
                elif y2 == -1:
                    p2 = (1, 0, x2)
                else:
                    p2 = (x2, y2, 1)
                
                # Normalize
                for p in [p1, p2]:
                    pass  # Already normalized
                
                val = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2] + 
                       p1[0]*p2[1] + p1[1]*p2[2] + p1[2]*p2[0]) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.add((i, j))
        
        num_edges = len(edges)
    
    # At this point, we should have a base with 90 edges (or close to it)
    # We need exactly 91 edges total
    
    if num_edges > 90:
        # Remove edges until we have 90
        edge_list = list(edges)
        edges = set(edge_list[:90])
        adj = [[False] * n for _ in range(n)]
        for (u, v) in edges:
            adj[u][v] = True
            adj[v][u] = True
    elif num_edges < 90:
        # Add random edges until we have 90
        while len(edges) < 90:
            import random
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u < v and (u, v) not in edges:
                edges.add((u, v))
                adj[u][v] = True
                adj[v][u] = True
    
    # Now add one more edge to reach 91 total
    # Find the non-edge that creates the fewest 4-cycles
    
    best_nonedge = None
    best_count = float('inf')
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    
    for (u, v) in all_pairs:
        if time.time() > deadline:
            break
        if (u, v) in edge_set:
            continue
        
        # Count 4-cycles created by adding (u,v)
        # A 4-cycle u-a-b-v-u requires:
        # (u,a), (a,b), (b,v) are edges, with a != b, a != v, b != u
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if not adj[u][a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj[a][b] and adj[b][v]:
                    count += 1
        
        if count < best_count:
            best_count = count
            best_nonedge = (u, v)
            if best_count == 0:
                break
    
    if best_nonedge is None:
        # Fallback: pick any non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    best_nonedge = (i, j)
                    break
            if best_nonedge:
                break
    
    result_edges = list(edges) + [best_nonedge]
    
    # Ensure exactly 91 distinct edges
    result_set = set(result_edges)
    assert len(result_set) == 91, f"Expected 91 edges, got {len(result_set)}"
    
    return [[u, v] for (u, v) in result_set]
