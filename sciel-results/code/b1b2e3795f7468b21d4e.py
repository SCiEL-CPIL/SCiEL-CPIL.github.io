"""Deterministic optimal solution: Fano plane on 7 vertices + 2 isolated."""

def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices
    # that is 3-chromatic and edge-minimal (7 edges).
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We use vertices 0-6 for the Fano plane and leave 7, 8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Verify: this is a valid 3-coloring where no edge is monochromatic
    # v0=0, v1=1, v2=2, v3=1, v4=2, v5=0, v6=1
    # Check each edge:
    # [0,1,2]: 0,1,2 - ok
    # [0,3,4]: 0,1,2 - ok
    # [0,5,6]: 0,0,1 - ok (not all same)
    # [1,3,5]: 1,1,0 - ok
    # [1,4,6]: 1,2,1 - ok
    # [2,3,6]: 2,1,1 - ok
    # [2,4,5]: 2,2,0 - ok
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}
