import time

def construct():
    deadline = time.time() + 115.0

    # --- Phase 1: Construct a C4-free base graph with known alpha < 11 ---
    # Use 10 disjoint triangles: n=30, C4-free, alpha=10.
    # This is a guaranteed valid solution.
    n_base = 30
    adj = [0] * n_base
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)

    # --- Phase 2: Try to add vertices greedily while maintaining C4-free and alpha < 11 ---
    # A new vertex v can be added if:
    # 1. For every pair of existing neighbors (u,w) of v, u and w share at most 1 common neighbor
    #    other than v (i.e., adding v doesn't create a C4).
    #    More precisely: for any two existing vertices u,w that are both neighbors of v,
    #    they must not have 2 common neighbors in the existing graph (since v would be a 3rd).
    #    Actually the C4 condition is: no two vertices have 2+ common neighbors.
    #    When we add v with neighbor set S, we need:
    #    - For any u in S: u and v share exactly |N(u) ∩ S| common neighbors. Must be <= 1.
    #      Wait, no. The condition is for ANY pair of vertices in the new graph.
    #    - For any pair (u, w) in the existing graph: their common neighbors in the new graph
    #      = common neighbors in old graph + (1 if both u and w are in S, else 0).
    #      So if u,w already have 1 common neighbor, they cannot both be in S.
    #    - For any u in S and v: common neighbors = |N(u) ∩ S|. Must be <= 1.
    #    - For any pair (u, w) both in S: common neighbors in new graph includes v plus
    #      any common neighbors in old graph. So |N(u) ∩ N(w) ∩ (old vertices)| + 1 <= 1
    #      meaning N(u) ∩ N(w) must be empty in the old graph.
    
    def can_add_vertex(adj, n, neighbor_mask):
        """Check if we can add a new vertex with the given neighbor set (as bitmask over 0..n-1)."""
        # Check 1: For any u in S, |N(u) ∩ S| <= 1
        for u in range(n):
            if neighbor_mask & (1 << u):
                common = adj[u] & neighbor_mask
                if common & (common - 1):  # more than one bit
                    return False
        
        # Check 2: For any pair (u,w) in existing graph, if they have 1 common neighbor already,
        # they can't both be in S
        # For any pair (u,w) both in S: |N(u) ∩ N(w)| in old graph must be 0
        for u in range(n):
            if not (neighbor_mask & (1 << u)):
                continue
            for w in range(u + 1, n):
                if not (neighbor_mask & (1 << w)):
                    continue
                # u and w are both neighbors of new vertex
                # Common neighbors of u and w in new graph = common in old + 1 (the new vertex)
                common_old = adj[u] & adj[w]
                if common_old:  # any common neighbor in old graph
                    return False
        
        # Check 3: For any u in S and any w not in S:
        # Common neighbors of u and w in new graph = common in old + (1 if w in S, else 0)
        # Since w is not in S, it's just common in old. No change. OK.
        
        # Check 4: For any u not in S and w not in S: no change. OK.
        # For any u in S and w = new vertex: covered by check 1.
        
        return True

    def has_independent_set_11(adj, n):
        """Check if there's an independent set of size 11. Returns True if exists."""
        if n < 11:
            return False
        
        # Branch and bound for maximum independent set
        # Sort by degree descending
        degrees = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -degrees[v])
        
        found = [False]
        
        def backtrack(idx, size, excluded):
            if found[0]:
                return
            if time.time() > deadline - 25:
                return
            if size >= 11:
                found[0] = True
                return
            if idx >= n:
                return
            # Upper bound: can we still reach 11?
            remaining = n - idx
            if size + remaining < 11:
                return
            
            v = order[idx]
            if excluded & (1 << v):
                backtrack(idx + 1, size, excluded)
                return
            
            # Option 1: exclude v
            backtrack(idx + 1, size, excluded)
            if found[0]:
                return
            
            # Option 2: include v
            new_excluded = excluded | (1 << v) | adj[v]
            backtrack(idx + 1, size + 1, new_excluded)
        
        backtrack(0, 0, 0)
        return found[0]

    # Try to add vertices one by one
    # We'll try adding vertices with various neighbor patterns
    # Strategy: try to add a vertex connected to exactly 1 vertex from each triangle
    # This maintains C4-free and keeps alpha at most 10 + 1 = 11... but we need alpha < 11.
    # If we connect to 1 vertex per triangle, the new vertex is not independent with any triangle vertex
    # it's connected to. But it could be independent with the other 2 vertices of each triangle.
    # Independent set: pick new vertex + 2 from each triangle = 1 + 20 = 21? No wait.
    # If new vertex is connected to vertex a_t in triangle t, then in an independent set containing
    # the new vertex, we can pick at most 1 from each triangle (the ones not connected).
    # Actually from triangle t, we can pick b_t or c_t (2 choices) since a_t is excluded.
    # So alpha could be 1 + 2*10 = 21. That's too high.
    
    # Better: connect the new vertex to ALL vertices in several triangles.
    # If connected to all 3 in triangle t, then no vertex from triangle t can be in an independent set with v.
    # Connect to all 3 in 10 triangles: alpha = 1 (just v) + 0 = 1. But that's a lot of edges.
    # Connect to all 3 in k triangles: alpha <= 1 + (10-k)*1 = 11-k. Need 11-k < 11, always true.
    # But we need to check C4-free.
    
    # Let's try a different approach: connect new vertex to 1 vertex in each of 10 triangles.
    # Then check C4-free and alpha.
    
    # Actually, let me try a smarter approach.
    # Start with the base and try to add vertices with carefully chosen neighborhoods.
    
    # Try adding vertex 30 connected to one vertex from each triangle
    # Connect to vertex 0 of each triangle: vertices 0, 3, 6, 9, ..., 27
    new_n = n_base + 1
    new_adj = [0] * new_n
    for i in range(n_base):
        new_adj[i] = adj[i]
    
    # Candidate: connect new vertex to {0, 3, 6, 9, 12, 15, 18, 21, 24, 27}
    neighbor_mask = 0
    for t in range(10):
        neighbor_mask |= (1 << (3 * t))
    
    if can_add_vertex(adj, n_base, neighbor_mask):
        # Check alpha
        for i in range(n_base):
            if neighbor_mask & (1 << i):
                new_adj[i] |= (1 << new_n)
        new_adj[new_n] = neighbor_mask
        
        if not has_independent_set_11(new_adj, new_n):
            n_base = new_n
            adj = new_adj
    
    # Try adding more vertices
    # Try vertex 31 connected to different pattern
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        new_n = n_base + 1
        new_adj = [0] * new_n
        for i in range(n_base):
            new_adj[i] = adj[i]
        
        # Try various neighbor patterns
        # Pattern: connect to 1 vertex from each triangle, but shift
        # For attempt k, connect to vertex (3*t + k%3) for t in 0..9
        neighbor_mask = 0
        for t in range(10):
            v = 3 * t + (attempt % 3)
            neighbor_mask |= (1 << v)
        
        if can_add_vertex(adj, n_base, neighbor_mask):
            for i in range(n_base):
                if neighbor_mask & (1 << i):
                    new_adj[i] |= (1 << new_n)
            new_adj[new_n] = neighbor_mask
            
            if not has_independent_set_11(new_adj, new_n):
                n_base = new_n
                adj = new_adj
    
    # Build edge list
    edges = []
    for u in range(n_base):
        for v in range(u + 1, n_base):
            if adj[u] & (1 << v):
                edges.append([u, v])
    
    # Final validation
    if n_base < 30:
        # Shouldn't happen, but fallback
        n_base = 30
        adj = [0] * n_base
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        edges = []
        for u in range(n_base):
            for v in range(u + 1, n_base):
                if adj[u] & (1 << v):
                    edges.append([u, v])
    
    return {'n': n_base, 'edges': edges}
