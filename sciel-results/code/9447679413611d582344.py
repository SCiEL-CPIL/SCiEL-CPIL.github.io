"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The Fano plane is a 3-uniform hypergraph on 7 vertices with 7 edges.
    # It is the unique 3-chromatic 3-uniform hypergraph on 7 vertices.
    # We need a 3-chromatic triple system on vertices 0..8 (9 vertices)
    # with the minimum number of edges.
    
    # A 3-chromatic hypergraph must have at least 4 edges (since 3 edges
    # on a 3-uniform system can always be 2-colored). Let's check if
    # we can do better than 7 edges.
    
    # Actually, let's think more carefully. The minimum number of edges
    # in a 3-chromatic 3-uniform hypergraph is known to be 7 (the Fano plane).
    # But we can have isolated vertices, so we just need 7 edges on some
    # subset of the 9 vertices.
    
    # Wait - can we have a 3-chromatic 3-uniform hypergraph with fewer than 7 edges?
    # A 2-coloring of a hypergraph assigns each vertex a color in {0,1} such that
    # no edge is monochromatic. So we need a hypergraph where every 2-coloring
    # has a monochromatic edge.
    
    # The Fano plane has 7 edges and is 3-chromatic. It's known that 7 is the
    # minimum for 3-uniform 3-chromatic hypergraphs.
    
    # Let's verify: the Fano plane edges are:
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring: assign colors so no edge is monochromatic
    # c[0]=0, c[1]=1, c[2]=2, c[3]=0, c[4]=1, c[5]=0, c[6]=2
    # Vertices 7 and 8 are isolated, assign color 0
    colors = [0, 1, 2, 0, 1, 0, 2, 0, 0]
    
    # Verify no edge is monochromatic in this coloring
    for e in edges:
        c = set(colors[v] for v in e)
        assert len(c) >= 2, f"Edge {e} is monochromatic"
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
