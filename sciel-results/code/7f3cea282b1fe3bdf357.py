import itertools

def is_2_colorable(edges, n_vertices):
    """
    Check if the hypergraph defined by edges on n_vertices is 2-colorable.
    Returns True if it is 2-colorable, False if it is not (i.e., 3-chromatic).
    """
    # Iterate over all 2^n colorings
    for mask in range(1 << n_vertices):
        # Build color array
        colors = [(mask >> i) & 1 for i in range(n_vertices)]
        valid = True
        for e in edges:
            if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                valid = False
                break
        if valid:
            return True
    return False

def find_min_3_chromatic():
    """
    Search for the minimum number of edges in a 3-chromatic 3-uniform hypergraph.
    """
    # We know Fano plane (7 edges) works.
    # Let's check if 6 edges suffice on 6 vertices.
    
    # Generate all possible 3-uniform edges on 6 vertices
    vertices = list(range(6))
    all_edges = list(itertools.combinations(vertices, 3))
    
    # Check subsets of size 6
    # Optimization: We can prune, but 38760 is small.
    for edges in itertools.combinations(all_edges, 6):
        if not is_2_colorable(edges, 6):
            return list(edges)
            
    # If no 6-edge solution on 6 vertices, check 7 vertices?
    # Or just fallback to Fano.
    # Let's try 7 edges on 7 vertices (Fano is known).
    # But maybe there is a 6-edge solution on 7 vertices?
    # Number of edges on 7 vertices: C(7,3) = 35.
    # C(35, 6) is ~1.6 million.
    # 2^7 = 128 checks.
    # 1.6M * 128 ~ 200M ops. Might be too slow in Python.
    # Let's stick to 6 vertices for 6 edges.
    # If not found, use Fano.
    
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    return fano_edges

def construct():
    edges_found = find_min_3_chromatic()
    
    # Pad edges to 9 vertices if necessary (they are already on 0..5 or 0..6)
    # If edges are on 0..5, vertices 6,7,8 are isolated.
    # If edges are on 0..6, vertices 7,8 are isolated.
    
    # Determine max vertex index used
    max_idx = -1
    for e in edges_found:
        for v in e:
            if v > max_idx:
                max_idx = v
                
    n_vertices_used = max_idx + 1
    
    # Find a valid 3-coloring
    # Brute force 3^n_vertices_used
    found_coloring = None
    for c in itertools.product([0, 1, 2], repeat=n_vertices_used):
        valid = True
        for e in edges_found:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            # Pad to 9 vertices
            found_coloring = list(c) + [0] * (9 - n_vertices_used)
            break
            
    if found_coloring is None:
        # Fallback
        found_coloring = [0, 1, 2, 0, 1, 2, 0, 0, 0]
        
    return {'edges': [list(e) for e in edges_found], 'colors': found_coloring}
