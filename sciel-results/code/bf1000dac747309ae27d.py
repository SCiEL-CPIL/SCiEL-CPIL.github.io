def construct():
    import time
    start_time = time.time()
    
    # Hoffman-Singleton graph construction
    # Vertices are (i, j) with i in 0..4, j in 0..9
    # Total 50 vertices.
    # We will index them 0..49
    # index = i * 10 + j
    
    n_total = 50
    n_target = 43
    
    edges = []
    
    # Define neighbors for each vertex
    # v(i, j) is adjacent to:
    # 1. (i, j+1)
    # 2. (i, j-1)
    # 3. (i+1, j+2)
    # 4. (i+1, j-2)
    # 5. (i-1, j+3)
    # 6. (i-1, j-3)
    # 7. (i+2, j+5) -- Wait, is this correct?
    
    # Let's check the standard adjacency for HS graph.
    # From literature:
    # The Hoffman-Singleton graph is the point graph of the generalized quadrangle GQ(2,2)? No.
    # It is a strongly regular graph (50, 7, 0, 1).
    
    # A known construction:
    # Vertices: Z_5 x Z_10
    # (i, j) ~ (i, j+1), (i, j-1)
    # (i, j) ~ (i+1, j+2), (i+1, j-2)
    # (i, j) ~ (i-1, j+3), (i-1, j-3)
    # (i, j) ~ (i+2, j+5) ? No, that's only 7 if we count carefully.
    # Let's count:
    # 2 from same i
    # 2 from i+1
    # 2 from i-1
    # 1 from i+2?
    # Total 7.
    
    # Let's verify if this is C4-free.
    # If this construction is correct, it is HS graph.
    
    def get_neighbors(i, j):
        neighbors = []
        # Same i
        neighbors.append((i, (j + 1) % 10))
        neighbors.append((i, (j - 1) % 10))
        # i+1
        neighbors.append(((i + 1) % 5, (j + 2) % 10))
        neighbors.append(((i + 1) % 5, (j - 2) % 10))
        # i-1
        neighbors.append(((i - 1) % 5, (j + 3) % 10))
        neighbors.append(((i - 1) % 5, (j - 3) % 10))
        # i+2 (or i-3)
        # To make it symmetric, usually the last one is unique.
        # Let's try (i+2, j+5)
        neighbors.append(((i + 2) % 5, (j + 5) % 10))
        return neighbors

    # Build adjacency list
    adj = {}
    for i in range(5):
        for j in range(10):
            v = i * 10 + j
            adj[v] = set()
            for ni, nj in get_neighbors(i, j):
                nv = ni * 10 + nj
                adj[v].add(nv)
    
    # Check if it's a valid HS graph (degree 7, symmetric)
    # And check for C4
    # For each vertex, check common neighbors of its neighbors?
    # Or just check if any pair has >1 common neighbor.
    
    # To be safe, we can just generate the edges and verify C4-freeness.
    # If it fails, we fall back to a smaller known graph.
    
    # Let's verify C4-freeness for the first 43 vertices.
    # Actually, if the whole graph is C4-free, any subgraph is.
    
    # Verification of C4-freeness:
    # For every pair of vertices u, v, count common neighbors.
    # If any count > 1, it's not C4-free.
    
    # This is O(n^2 * degree) which is small.
    
    is_c4_free = True
    for u in range(50):
        for v in range(u + 1, 50):
            common = adj[u] & adj[v]
            if len(common) > 1:
                is_c4_free = False
                break
        if not is_c4_free:
            break
            
    if not is_c4_free:
        # Fallback: Use the seed program's logic but maybe extend it?
        # Or just return the seed result.
        # But let's assume the construction is correct.
        # If it's not, we return n=30.
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Construct edges for the first n_target vertices
    # We need to ensure the induced subgraph is connected? No, just simple graph.
    # We take vertices 0 to n_target-1.
    
    # Filter edges to those within 0..n_target-1
    for u in range(n_target):
        for v in adj[u]:
            if v < n_target and u < v:
                edges.append([u, v])
                
    return {'n': n_target, 'edges': edges}
