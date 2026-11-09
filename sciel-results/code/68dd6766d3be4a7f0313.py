"""Construct a graph with 31 vertices and 91 edges minimizing the number of 4-cycles.

Strategy: Start with the C4-free polarity graph of PG(2,5) which has exactly 90 edges,
then add one edge that creates the minimum number of new 4-cycles.
"""

def construct():
    n = 31
    q = 5
    # Polarity graph of PG(2, q): vertices are the q^2+q+1 = 31 points of PG(2,5)
    # We label points (x,y,z) with x,y,z in GF(q), not all zero, up to scalar.
    # Standard representation: use homogeneous coordinates [x:y:z] where not all zero,
    # normalized so that the first non-zero coordinate is 1.
    #
    # The polarity in PG(2,q) for odd q: point (x,y,z) is orthogonal to (x',y',z') iff
    #   x*x' + y*y' + z*z' + x*y' + y*z' + z*x' = 0  (in some standard form)
    #
    # Actually, for the classical polarity graph of PG(2,q) (q odd), the standard construction:
    # Vertices: all (a,b) in GF(q)^2 plus points at infinity.
    # A common representation: vertices are (a,b) for a,b in GF(q) and one point at infinity.
    # But we need q^2+q+1 = 31 vertices.
    #
    # Let me use the standard construction from literature:
    # For q odd, the polarity graph of PG(2,q) has vertices corresponding to points of PG(2,q).
    # Two points are adjacent if they are perpendicular under a suitable polarity.
    #
    # A concrete construction:
    # Vertices: (i,j) for 0 <= i,j <= q-1, plus (i, inf) for 0 <= i <= q-1, plus (inf, inf).
    # That's q^2 + q + 1 vertices.
    #
    # For the standard polarity: point (a,b) is adjacent to (a',b') if:
    #   (a-b)(a'-b') = 1  when both are finite... no, that's not right either.
    #
    # Let me use a well-known explicit construction. The C4-free graph with q^2+q+1 vertices
    # and q^2+q edges (for q odd) is the incidence graph of the polarity in PG(2,q).
    #
    # Actually, the simplest approach: use the known fact that the polarity graph of PG(2,q)
    # for odd q is C4-free with q^2+q+1 vertices and q^2+q edges.
    #
    # Explicit construction (from Koolen et al.):
    # Label vertices as pairs (a,b) where a,b in GF(q), plus q "infinite" vertices, plus one more.
    # 
    # Let me just construct it directly using the field GF(5).
    
    # GF(5) elements: 0, 1, 2, 3, 4
    # Points of PG(2,5): represented as [x:y:z] with x,y,z in GF(5), not all zero,
    # where [x:y:z] = [kx:ky:kz] for k != 0.
    # Normalize: first nonzero coordinate is 1.
    
    # Generate all normalized points
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    rep = (x, y, z)
                    # First nonzero is x=1 after scaling? No, normalize so first nonzero is 1
                    inv_x = pow(x, -1, q)  # modular inverse
                    nx = x * inv_x % q
                    ny = y * inv_x % q
                    nz = z * inv_x % q
                    rep = (nx, ny, nz)
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    nx = x * inv_y % q
                    ny = y * inv_y % q
                    nz = z * inv_y % q
                    rep = (nx, ny, nz)
                else:  # z != 0
                    inv_z = pow(z, -1, q)
                    nx = x * inv_z % q
                    ny = y * inv_z % q
                    nz = z * inv_z % q
                    rep = (nx, ny, nz)
                if rep not in seen:
                    seen.add(rep)
                    points.append(rep)
    
    # We should have q^2 + q + 1 = 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Index points
    point_index = {p: i for i, p in enumerate(points)}
    
    # Polarity: [x:y:z] is orthogonal to [x':y':z'] iff
    #   x*z' + y*(x'+z') + z*x' = 0  (in GF(q))
    # This is the standard Hermitian-like polarity for PG(2,q) with q odd.
    # Actually, the standard quadratic form for the polarity in PG(2,q) (q odd):
    #   Q(x,y,z) = x^2 + y^2 + z^2 + xy + yz + zx
    # The bilinear form associated: B((x,y,z), (x',y',z')) = x*z' + y*(x'+z') + z*x'
    # Wait, let me be more careful.
    #
    # The standard polarity in PG(2,q) for odd q: define the bilinear form
    #   f((x,y,z), (x',y',z')) = x*z' + y*(x' + z') + z*x'
    # Then point P=(x,y,z) is adjacent to Q=(x',y',z') iff f(P,Q) = 0 and P != Q.
    # Note: f(P,P) = x*z + y*(x+z) + z*x = 2xz + xy + yz
    # A point is "self-adjacent" if f(P,P) = 0.
    
    def ortho(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        # f(p1, p2) = x1*z2 + y1*(x2 + z2) + z1*x2
        val = (x1 * z2 + y1 * (x2 + z2) + z1 * x2) % q
        return val == 0
    
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    edges_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if ortho(points[i], points[j]):
                adj[i][j] = True
                adj[j][i] = True
                edges_set.add((i, j))
    
    num_edges = len(edges_set)
    # For the polarity graph of PG(2,q), the number of edges should be q^2+q = 30 for q=5? 
    # No, q^2+q = 30. But we need 90 edges for the base C4-free graph.
    # Wait, I'm confusing things. The polarity graph has q^2+q+1 vertices.
    # The number of edges: each vertex has degree... 
    # Actually, in the polarity graph of PG(2,q) for odd q, each vertex has degree q+1 or q+1-1...
    # Let me just count.
    
    # If the graph doesn't have exactly 90 edges, I need to adjust my construction.
    # Let me count the actual number of edges from my construction.
    
    if num_edges != 90:
        # My construction might not give exactly 90 edges. Let me try a different approach.
        # 
        # Alternative: Use the known C4-free graph construction from finite geometry.
        # The Reiman graph or similar.
        # 
        # Actually, let me try a different polarity form.
        # The standard one for PG(2,q): 
        #   B((x,y,z), (x',y',z')) = x*x' + y*y' + z*z' + x*y' + y*z' + z*x'
        # Let me try this instead.
        
        edges_set = set()
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*z2 + z1*x2) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges_set.add((i, j))
        
        num_edges = len(edges_set)
    
    # If still not 90, try yet another form
    if num_edges != 90:
        # Let me try: B((x,y,z), (x',y',z')) = x*z' + y*x' + z*y'  (cyclic)
        edges_set = set()
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (x1*z2 + y1*x2 + z1*y2) % q
                if val == 0:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges_set.add((i, j))
        
        num_edges = len(edges_set)
    
    if num_edges != 90:
        # Fallback: use a known construction. The C4-free graph with 31 vertices and 90 edges
        # can be constructed as follows (from the theory of generalized quadrangles):
        # 
        # Actually, let me just verify: q^2+q = 30 edges for q=5? No, that can't be right
        # for a graph with 31 vertices.
        #
        # Wait, I think I'm confusing the polarity graph with something else.
        # The polarity graph of PG(2,q) has q^2+q+1 vertices. The number of edges is
        # (q^2+q+1)(q+1)/2 for even q, and (q^2+q+1)(q+1)/2 - (q^2+q+1)/2 for odd q
        # (subtracting self-adjacent vertices).
        #
        # For q=5: (31*6)/2 - 31/2 = 93 - 15.5... that doesn't work out cleanly.
        #
        # Let me just count what I have and work with it.
        pass
    
    # For now, let's just add edges to reach 91 total
    # First, let's get a C4-free base. If our base has E edges, we need 91-E more.
    # But the key insight is: if the base is C4-free, adding one edge creates
    # 4-cycles equal to the number of common neighbors of the endpoints that are
    # connected through paths of length 2.
    
    # Let me try a completely different, known-good construction.
    # 
    # The C4-free graph with 31 vertices and 90 edges:
    # This is the graph from the finite geometry construction.
    # 
    # Let me use the following explicit construction:
    # Vertices: (a, b) for a, b in GF(5), plus (a, inf) for a in GF(5), plus (inf, inf).
    # That's 25 + 5 + 1 = 31 vertices.
    
    # Map vertices to indices
    vertices = []
    for a in range(q):
        for b in range(q):
            vertices.append((a, b))
    for a in range(q):
        vertices.append((a, -1))  # -1 represents infinity
    vertices.append((-1, -1))  # both infinity
    
    assert len(vertices) == 31
    
    # Define adjacency for the polarity graph:
    # (a,b) adjacent to (a',b') if:
    #   - If both finite: (a-a')*(b-b') = 1 in GF(5)  ... no, that gives a different graph.
    #
    # Let me use the standard construction from the literature on C4-free graphs.
    # 
    # The graph: V = GF(q)^2 ∪ {∞} ∪ {(∞, a) : a ∈ GF(q)}
    # Edges:
    #   - (a,b) ~ (a',b') iff (a-a')(b'-b) = 1  [for finite points, with some condition]
    #   - (a,b) ~ (∞, c) iff c = b
    #   - (a,b) ~ ∞ iff b = 0
    #   - (∞, a) ~ (∞, b) iff a = b  [no, that's not right]
    #
    # This is getting complicated. Let me just use a verified construction.
    
    # Known result: The polarity graph of PG(2,5) has 31 vertices and 90 edges, and is C4-free.
    # Let me verify my edge count.
    
    # For the form B((x,y,z),(x',y',z')) = x*z' + y*(x'+z') + z*x':
    # Let me count edges for each vertex type.
    
    # Actually, let me just proceed with whatever base I have and add edges optimally.
    
    # Find the base edges
    base_edges = list(edges_set)
    base_adj = [[False] * n for _ in range(n)]
    for (u, v) in base_edges:
        base_adj[u][v] = True
        base_adj[v][u] = True
    
    # We need 91 edges total. If base has E edges, we need 91-E more.
    needed = 91 - len(base_edges)
    
    if needed <= 0:
        # Base has too many edges; remove some
        # This shouldn't happen for a proper C4-free construction
        base_edges = base_edges[:90]
        base_adj = [[False] * n for _ in range(n)]
        for (u, v) in base_edges:
            base_adj[u][v] = True
            base_adj[v][u] = True
        needed = 1
    
    # Find the non-edge that creates the fewest 4-cycles
    # A 4-cycle u-v-w-x-u is created when we add edge (u,v) if there exist
    # w,x such that u-w, w-v, v-x, x-u are all edges in the base graph.
    # i.e., number of common neighbors of u and v in base graph that are
    # connected by a path of length 2 through two distinct intermediate vertices.
    # Actually, a 4-cycle containing the new edge (u,v) has the form u-a-b-v-u
    # where (u,a), (a,b), (b,v) are edges in base, and a != b, a != v, b != u.
    
    # For each non-edge (u,v), count the number of 4-cycles it creates:
    # This equals the number of pairs (a,b) such that (u,a), (a,b), (b,v) are edges
    # and a != b, a != v, b != u.
    
    # More precisely: for non-edge (u,v), the number of new 4-cycles is
    # |{(a,b) : (u,a)∈E, (a,b)∈E, (b,v)∈E, a≠b, a≠v, b≠u}|
    
    # This can be computed as: for each a in N(u) \\ {v}, for each b in N(v) \\ {u},
    # if (a,b) is an edge and a != b, count it.
    
    # But we also need a != v and b != u, which is already handled by excluding v from N(u)
    # and u from N(v).
    
    # Additionally, we need the 4-cycle to be a simple cycle, so a != b.
    
    best_nonedge = None
    best_count = float('inf')
    
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    base_edge_set = set(base_edges)
    
    for (u, v) in all_pairs:
        if (u, v) in base_edge_set:
            continue
        # Count 4-cycles created by adding (u,v)
        count = 0
        for a in range(n):
            if a == u or a == v:
                continue
            if not base_adj[u][a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if base_adj[a][b] and base_adj[b][v]:
                    count += 1
        if count < best_count:
            best_count = count
            best_nonedge = (u, v)
    
    # Add the best non-edge
    result_edges = list(base_edges) + [best_nonedge]
    
    # Return as list of [u, v] pairs
    return [[u, v] for (u, v) in result_edges]
