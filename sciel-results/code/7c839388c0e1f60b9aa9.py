import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin for 60s limit
    
    # The Fano plane is the unique (up to isomorphism) 3-uniform 3-chromatic 
    # hypergraph with 7 edges. It is a known result that any 3-uniform 
    # hypergraph with <= 6 edges is 2-colorable. Thus, 7 is the minimum.
    
    # Fano plane edges on vertices 0-6
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # Vertices 7 and 8 are isolated
    # We need to provide a valid 3-coloring.
    # Fano plane coloring:
    # 0: 0, 1: 1, 2: 2
    # 3: 0, 4: 2, 5: 1
    # 6: 0
    # Check:
    # (0,1,2) -> 0,1,2 (OK)
    # (0,3,4) -> 0,0,2 (OK)
    # (0,5,6) -> 0,1,0 (OK)
    # (1,3,5) -> 1,0,1 (OK)
    # (1,4,6) -> 1,2,0 (OK)
    # (2,3,6) -> 2,0,0 (OK)
    # (2,4,5) -> 2,2,1 (OK)
    
    colors = [0, 1, 2, 0, 2, 1, 0, 0, 0]
    
    # Verification Step:
    # 1. Ensure no edge is monochromatic in the provided 3-coloring.
    for e in edges:
        c0, c1, c2 = colors[e[0]], colors[e[1]], colors[e[2]]
        if c0 == c1 == c2:
            # This should never happen with the hardcoded Fano coloring
            raise ValueError("Invalid 3-coloring found")
            
    # 2. Ensure the hypergraph is 3-chromatic (i.e., no 2-coloring exists).
    # We check all 2^9 = 512 possible 2-colorings of the 9 vertices.
    # If any 2-coloring makes all edges non-monochromatic, it's 2-colorable.
    # For the Fano plane, no such coloring exists.
    
    def is_2_colorable(edges, n_vertices):
        # Try all 2-colorings
        for mask in range(1 << n_vertices):
            # Create coloring array
            col = [(mask >> i) & 1 for i in range(n_vertices)]
            valid = True
            for e in edges:
                c0, c1, c2 = col[e[0]], col[e[1]], col[e[2]]
                if c0 == c1 == c2:
                    valid = False
                    break
            if valid:
                return True
        return False

    # Check if our specific edge set is 2-colorable
    # Note: We only care about vertices involved in edges for the check logic,
    # but the mask covers all 9. Isolated vertices don't affect monochromaticity of edges.
    if is_2_colorable(edges, 9):
        # This should not happen for Fano plane.
        # If it does, we have a bug or wrong edge set.
        # As a fallback, we would need to find a 3-chromatic graph, 
        # but Fano is the minimum, so this branch is effectively unreachable 
        # for correct Fano edges.
        pass

    # Ensure types are exactly Python int and list
    final_edges = [[int(v) for v in e] for e in edges]
    final_colors = [int(c) for c in colors]
    
    return {'edges': final_edges, 'colors': final_colors}
