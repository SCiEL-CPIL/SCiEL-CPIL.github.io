def construct():
    import itertools
    import time

    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def has_independent_set_of_size_k(n, adj, k):
        # Branch and bound to check if alpha >= k
        # We want to know if alpha < 10, so we check if there is an independent set of size 10.
        # If yes, invalid. If no, valid.
        
        def dfs(remaining, current_set):
            if len(current_set) == k:
                return True
            if not remaining:
                return False
            
            # Upper bound: if len(current_set) + len(remaining) < k, prune
            if len(current_set) + len(remaining) < k:
                return False
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with min degree in remaining?
            # Or just pick the first one.
            v = next(iter(remaining))
            
            # Option 1: Include v
            new_remaining_include = {u for u in remaining if u != v and not adj[v][u]}
            if dfs(new_remaining_include, current_set | {v}):
                return True
            
            # Option 2: Exclude v
            new_remaining_exclude = remaining - {v}
            if dfs(new_remaining_exclude, current_set):
                return True
                
            return False

        return dfs(set(range(n)), set())

    def get_graph(n, steps):
        edges = []
        for i in range(n):
            for s in steps:
                j = (i + s) % n
                if i < j:
                    edges.append([i, j])
        return edges

    def try_circulant(n, steps):
        edges = get_graph(n, steps)
        if not is_triangle_free(n, edges):
            return None
        # Build adj
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        if has_independent_set_of_size_k(n, adj, 10):
            return None
        return {'n': n, 'edges': edges}

    # Try n from 40 down to 30
    for n in range(40, 29, -1):
        # Try some step sets
        # For triangle-free, steps must not contain a, b, a+b
        # Simple steps: [1], [2], [3]...
        # For n=40, [1] is C40, alpha=20.
        # We need denser graphs.
        # Try steps like [1, 2, 4, 8]?
        # Let's try a few random-looking sets
        candidate_steps = [
            [1, 2, 4, 8, 16],
            [1, 3, 5, 7],
            [1, 2, 3, 5, 7],
            [1, 4, 6, 9, 11],
            [1, 2, 5, 10, 15]
        ]
        
        for steps in candidate_steps:
            # Filter steps to be < n/2 and unique
            valid_steps = [s % n for s in steps if s % n != 0]
            valid_steps = list(set(valid_steps))
            if not valid_steps:
                continue
            
            res = try_circulant(n, valid_steps)
            if res:
                return res

    # Fallback to a known valid graph
    # The Clebsch graph is 16 vertices, alpha=5
    # We can just return the seed's fallback or a simple valid graph
    # Let's return a valid graph on n=16 (Clebsch)
    # Or just the seed's n=26 if it was valid?
    # The seed's n=26 circulant [1] is C26, alpha=13. Invalid.
    
    # Let's construct a valid graph on n=16 (Clebsch)
    # Clebsch graph edges
    clebsch_edges = []
    # Clebsch is the complement of the 4x4 grid? No.
    # It is the folded 5-cube.
    # Vertices 0..15.
    # i ~ j if popcount(i xor j) == 1 or popcount(i xor j) == 4?
    # No, Clebsch is triangle-free.
    # Let's just use a known valid small graph.
    # Actually, the task wants to maximize n.
    # If we can't find n>30, we should return the best we have.
    # But an invalid graph scores 0.
    # So we must return a VALID graph.
    
    # Let's try to generate a valid graph on n=16 using a simple method
    # C5 x C5? No.
    
    # I will return a valid graph on n=16 (Clebsch)
    # Vertices: 0..15
    # Edges: (i, i+1) mod 16? No, that's C16, alpha=8.
    
    # Let's use the Schrijver graph S(5,2) which is 16 vertices?
    # Or just the Clebsch graph.
    
    # Clebsch graph adjacency:
    # Two vertices are adjacent if their Hamming distance is 1 or 4?
    # No, that's the 4-cube complement?
    
    # Let's use a simple valid graph:
    # The 5-cycle C5 has alpha=2.
    # Disjoint union of C5s?
    # 3 * C5 = 15 vertices, alpha=6.
    # 4 * C5 = 20 vertices, alpha=8.
    # 5 * C5 = 25 vertices, alpha=10. (Invalid, alpha=10)
    # 4 * C5 + 1 isolated? No, isolated increases alpha.
    
    # 4 * C5 is 20 vertices, alpha=8. Valid.
    # Can we do better?
    # Clebsch (16, alpha=5).
    # 2 * Clebsch = 32 vertices, alpha=10. Invalid.
    # Clebsch + C5 = 21 vertices, alpha=7.
    
    # Let's return 4 * C5 (n=20, alpha=8) as a safe fallback.
    edges = []
    for c in range(4):
        offset = c * 5
        for i in range(5):
            u = offset + i
            v = offset + (i + 1) % 5
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    return {'n': 20, 'edges': edges}
