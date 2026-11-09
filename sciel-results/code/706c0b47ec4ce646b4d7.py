def construct():
    # The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph
    # with 7 edges that is 3-chromatic. It is known to be the minimum
    # 3-uniform 3-chromatic hypergraph (Erdos-Lovasz, 1975).
    # We embed it on 9 vertices (7,8 isolated) to satisfy the 9-vertex constraint.
    
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring of the Fano plane
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verification 1: The provided 3-coloring must be valid
    # (Every edge must be non-monochromatic)
    for e in edges:
        c = {colors[v] for v in e}
        if len(c) < 2:
            raise ValueError(f"Edge {e} is monochromatic under provided colors")
    
    # Verification 2: The hypergraph must NOT be 2-colorable.
    # We use a backtracking search to determine if a 2-coloring exists.
    n_vertices = 9
    # Mapping vertex index to list of edges it belongs to, for pruning
    vertex_edges = [[] for _ in range(n_vertices)]
    for idx, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(idx)
    
    # Backtracking state
    color_assignment = [-1] * n_vertices
    
    def is_2_colorable():
        """Returns True if a 2-coloring exists."""
        # Recursive helper
        def backtrack(vertex_idx):
            if vertex_idx == n_vertices:
                # All vertices colored. Check if all edges are satisfied (non-monochromatic)
                for e in edges:
                    c = {color_assignment[v] for v in e}
                    if len(c) == 1:
                        return False
                return True
            
            # Try color 0
            color_assignment[vertex_idx] = 0
            
            valid = True
            for e_idx in vertex_edges[vertex_idx]:
                e = edges[e_idx]
                # Check if this edge is now monochromatic
                # Only possible if all vertices in e are colored
                all_colored = all(color_assignment[v] != -1 for v in e)
                if all_colored:
                    c = {color_assignment[v] for v in e}
                    if len(c) == 1:
                        valid = False
                        break
            
            if valid:
                if backtrack(vertex_idx + 1):
                    return True
            
            # Try color 1
            color_assignment[vertex_idx] = 1
            valid = True
            for e_idx in vertex_edges[vertex_idx]:
                e = edges[e_idx]
                all_colored = all(color_assignment[v] != -1 for v in e)
                if all_colored:
                    c = {color_assignment[v] for v in e}
                    if len(c) == 1:
                        valid = False
                        break
            
            if valid:
                if backtrack(vertex_idx + 1):
                    return True
            
            color_assignment[vertex_idx] = -1
            return False
        
        return backtrack(0)
    
    if is_2_colorable():
        raise ValueError("Fano plane appears to be 2-colorable — logic error")
    
    result = {
        'edges': list(edges),
        'colors': list(colors)
    }
    
    # Final type assertions
    assert isinstance(result['edges'], list)
    assert len(result['edges']) >= 1
    for e in result['edges']:
        assert isinstance(e, tuple)
        for v in e:
            assert isinstance(v, int) and not isinstance(v, bool)
            assert 0 <= v <= 8
    assert isinstance(result['colors'], list)
    assert len(result['colors']) == 9
    for c in result['colors']:
        assert isinstance(c, int) and not isinstance(c, bool)
        assert c in (0, 1, 2)
    
    return result
