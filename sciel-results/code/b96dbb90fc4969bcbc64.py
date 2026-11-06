def decompose(edges):
    if not edges:
        return []
    m = len(edges)
    if m == 0:
        return []
    
    # Build cost matrix
    # cost[i][j] = 0 if edges[i] and edges[j] share a vertex, else BIG
    BIG = 10**9
    cost = np.full((m, m), BIG, dtype=np.int64)
    
    # Precompute vertex sets for edges
    edge_vertices = [set(e) for e in edges]
    
    for i in range(m):
        for j in range(m):
            if i != j and edge_vertices[i] & edge_vertices[j]:
                cost[i][j] = 0
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    next_edge = [-1] * m
    prev_edge = [-1] * m
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] == 0:
            next_edge[r] = c
            prev_edge[c] = r
            
    paths = []
    for i in range(m):
        if prev_edge[i] == -1:
            path_edges = []
            cur = i
            while cur != -1:
                path_edges.append(cur)
                cur = next_edge[cur]
            
            # Convert to vertex list
            # We need to determine the order of vertices.
            # Since we ignored direction, we just know the sequence of edges.
            # We must orient them to form a valid path.
            # e0, e1, ... ek
            # v0 - e0 - v1 - e1 - v2 ...
            # We can reconstruct the path by walking.
            # Start with e0. It has 2 vertices.
            # Pick one as start.
            # Then the next edge must share the current end.
            
            if not path_edges:
                continue
                
            verts = []
            # Determine the path order
            # This is a bit tricky with undirected edges in a sequence.
            # We can just take the union of vertices? No, we need the sequence.
            
            # Let's try to orient.
            # current_vertex = None
            # for e in path_edges:
            #    if current_vertex is None:
            #       pick arbitrary start
            #    else:
            #       must start at current_vertex
            #       end at the other vertex
            
            current_vertex = None
            path_vertices = []
            for e_idx in path_edges:
                u, v = edges[e_idx]
                if current_vertex is None:
                    # Start of path
                    path_vertices.append(u)
                    current_vertex = v
                else:
                    if u == current_vertex:
                        path_vertices.append(u)
                        current_vertex = v
                    elif v == current_vertex:
                        path_vertices.append(v)
                        current_vertex = u
                    else:
                        # Disconnected? Should not happen if matching is valid.
                        # But if it does, it's a new path.
                        path_vertices.append(u)
                        current_vertex = v
            paths.append(path_vertices)
            
    return paths
