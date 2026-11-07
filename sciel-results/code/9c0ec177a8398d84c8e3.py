"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a 3-uniform hypergraph on 9 vertices with chromatic number 3
    and the minimum possible number of edges (7).
    
    The Fano plane is the unique 3-chromatic 3-uniform hypergraph with 7 edges.
    We embed it in the 9-vertex space {0..8} using vertices 0..6.
    Vertices 7 and 8 are isolated and can have any color.
    """
    
    # Fano plane edges (7 edges on vertices 0-6)
    # These are the lines of the projective plane of order 2.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6:
    # We need to assign colors 0, 1, 2 to vertices 0..6 such that
    # no edge is monochromatic.
    # 
    # Let's verify the coloring:
    # v0=0, v1=1, v2=2 -> Edge [0,1,2] has colors {0,1,2} (OK)
    # v0=0, v3=1, v4=2 -> Edge [0,3,4] has colors {0,1,2} (OK)
    # v0=0, v5=0, v6=1 -> Edge [0,5,6] has colors {0,1} (OK)
    # v1=1, v3=1, v5=0 -> Edge [1,3,5] has colors {0,1} (OK)
    # v1=1, v4=2, v6=1 -> Edge [1,4,6] has colors {1,2} (OK)
    # v2=2, v3=1, v6=1 -> Edge [2,3,6] has colors {1,2} (OK)
    # v2=2, v4=2, v5=0 -> Edge [2,4,5] has colors {0,2} (OK)
    
    # Note: The specific coloring above works.
    # Let's double check v5=0, v6=1.
    # Edge [0,5,6]: 0,0,1 -> Not mono.
    # Edge [1,3,5]: 1,1,0 -> Not mono.
    # Edge [2,4,5]: 2,2,0 -> Not mono.
    # Edge [0,5,6] is OK.
    
    # Wait, let's re-verify the standard Fano coloring.
    # Vertices 0..6.
    # Colors:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 1
    # 4: 2
    # 5: 0
    # 6: 1
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    # Ensure types are strictly Python int and list
    # The evaluator rejects NumPy integers and bools.
    edges = [[int(a), int(b), int(c)] for a, b, c in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}

# EVOLVE-BLOCK-END
