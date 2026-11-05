"""C5-free graphs without independent 9-sets: structural design using Petersen components."""
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C5-free graph with no independent set of size 9.
    Returns a dictionary with 'n' (number of vertices) and 'edges' (list of edge pairs).
    
    Strategy:
    Use 8 disjoint K4s.
    - n = 32
    - C5-free: K4 has no C5. Disjoint union of C5-free graphs is C5-free.
    - Independence number: Each K4 contributes at most 1 to any independent set.
      Total alpha = 8 * 1 = 8 < 9.
    
    This achieves n=32, which is the best known lower bound for R(C5, K9).
    Attempts to extend to n=33 within this family fail due to C5 constraints
    on the added vertex's connections to the K4 components.
    """
    n = 32
    edges = []
    
    # Construct 8 disjoint K4s
    # Vertices 0-3 form K4_0, 4-7 form K4_1, ..., 28-31 form K4_7
    for block in range(8):
        base = block * 4
        # Add all edges within the K4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
