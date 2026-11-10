"""Program for dense odd-order graphs avoiding equal-degree path endpoints."""

# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    
    best_edges = []
    max_edge_count = -1
    
    # Iterate over all possible threshold graphs
    # A threshold graph on n vertices is defined by a binary sequence of length n-1
    # b_1, ..., b_{n-1} where b_i=1 means vertex i is dominating (connected to all previous),
    # b_i=0 means vertex i is isolated (connected to no previous).
    # Vertex 0 is the first vertex.
    
    # Pre-compute bit masks for speed? Or just loop.
    # 2^14 = 16384 iterations.
    
    for mask in range(1 << (n - 1)):
        adj = [[False] * n for _ in range(n)]
        
        # Build the threshold graph
        # Vertex 0 is isolated initially.
        for i in range(1, n):
            # Check if bit (i-1) is set
            if (mask >> (i - 1)) & 1:
                # Dominating: connect to all previous vertices 0..i-1
                for j in range(i):
                    adj[i][j] = True
                    adj[j][i] = True
            # If 0, isolated: no connections to previous.
        
        # Calculate degrees
        deg = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    deg[i] += 1
        
        # Check validity
        # Group vertices by degree
        from collections import defaultdict
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[deg[i]].append(i)
        
        valid = True
        # For each group of vertices with the same degree, check pairs
        for d, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            # Check all pairs in this group
            for idx_s in range(len(vertices)):
                for idx_t in range(idx_s + 1, len(vertices)):
                    s = vertices[idx_s]
                    t = vertices[idx_t]
                    
                    # Check if there is a simple path of length 3 between s and t
                    # Path: s - u - v - t
                    # s != t (guaranteed by pair selection)
                    # u != v, u != t, v != s (guaranteed by simple path definition in check)
                    
                    # Case 1: s and t are adjacent
                    if adj[s][t]:
                        # Need u in N(s)\{t} and v in N(t)\{s} such that adj[u][v]
                        # Iterate u in N(s)
                        found = False
                        for u in range(n):
                            if u == t or not adj[s][u]:
                                continue
                            # u is a candidate for the first step
                            # Now we need v in N(t)\{s} such that adj[u][v]
                            # And v != u (guaranteed if adj[u][v] and no loops, but u!=v is required for simple path)
                            # Also v != s is required.
                            for v in range(n):
                                if v == s or v == u or not adj[t][v] or not adj[u][v]:
                                    continue
                                # Path s-u-v-t is simple and valid
                                found = True
                                break
                            if found:
                                break
                        if found:
                            valid = False
                            break
                    else:
                        # Case 2: s and t are not adjacent
                        # Need u in N(s) and v in N(t) such that adj[u][v] and u != v
                        found = False
                        for u in range(n):
                            if not adj[s][u]:
                                continue
                            for v in range(n):
                                if v == u or not adj[t][v] or not adj[u][v]:
                                    continue
                                found = True
                                break
                            if found:
                                break
                        if found:
                            valid = False
                            break
                    if not valid:
                        break
            if not valid:
                break
        
        if valid:
            # Count edges
            edge_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edge_count += 1
            
            if edge_count > max_edge_count:
                max_edge_count = edge_count
                # Build edge list
                best_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            best_edges.append([i, j])
    
    # If no valid threshold graph found (unlikely, star is one), fallback to empty or star
    if max_edge_count == -1:
        # Star graph: 0 connected to 1..n-1
        # Check if star is valid.
        # deg(0)=n-1, deg(i)=1.
        # Groups: {0}, {1, ..., n-1}.
        # In group {1..n-1}, all have deg 1.
        # Pick s=1, t=2.
        # Path 1-0-3-2?
        # 1-0 (edge), 0-3 (edge), 3-2 (no edge).
        # In star, only 0 is connected to others.
        # Path s-u-v-t.
        # u must be 0 (only neighbor of s).
        # v must be neighbor of 0 and t.
        # v must be t? No, v is intermediate.
        # If u=0, then path is s-0-v-t.
        # s is leaf, 0 is center.
        # v must be neighbor of 0.
        # t is neighbor of v.
        # In star, neighbors of 0 are all leaves.
        # So v is a leaf.
        # t must be neighbor of v.
        # v is leaf, only neighbor is 0.
        # So t must be 0.
        # But s != t. s is leaf, t is 0.
        # Path s-0-v-0? No, simple path.
        # So no 3-path between two leaves in a star?
        # Distance between two leaves is 2.
        # So star is valid.
        
        best_edges = [[0, i] for i in range(1, n)]
        max_edge_count = n - 1

    return best_edges

# EVOLVE-BLOCK-END
