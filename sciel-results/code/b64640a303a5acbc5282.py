import time

def construct():
    start = time.time()
    deadline = start + 55.0

    # The Fano plane is the unique (up to isomorphism) 3-chromatic 3-uniform
    # hypergraph with 7 edges on 7 vertices. It is well-established that no
    # 3-uniform hypergraph with 6 or fewer edges is 3-chromatic.
    #
    # Therefore, the minimum number of edges for a 3-chromatic 3-uniform
    # hypergraph is 7, achieved by the Fano plane.
    #
    # We embed the Fano plane on vertices 0..6 (vertices 7,8 are isolated).
    
    fano_edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Valid 3-coloring for the Fano plane vertices
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Verify coloring validity
    for e in fano_edges:
        c = set(fano_colors[v] for v in e)
        if len(c) == 1:
            raise ValueError("Invalid coloring: monochromatic edge found")
            
    # Verify 3-chromaticity (not 2-colorable)
    def is_2_colorable(edges, n):
        for mask in range(1 << n):
            ok = True
            for e in edges:
                first_color = (mask >> e[0]) & 1
                all_same = True
                for v in e[1:]:
                    if ((mask >> v) & 1) != first_color:
                        all_same = False
                        break
                if all_same:
                    ok = False
                    break
            if ok:
                return True
        return False
    
    if is_2_colorable(fano_edges, 9):
        raise ValueError("Hypergraph is 2-colorable, invalid for 3-chromatic requirement")
        
    # Return the result with strict types
    return {
        "edges": [list(e) for e in fano_edges], 
        "colors": list(fano_colors)
    }
