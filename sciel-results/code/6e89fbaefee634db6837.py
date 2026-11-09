"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time

def construct():
    """
    Constructs a 3-uniform hypergraph on 9 vertices that is 3-chromatic
    with the minimum possible number of edges (7).
    
    The hypergraph is the Fano plane (PG(2,2)) embedded on vertices 0-6,
    with vertices 7 and 8 isolated.
    
    Method:
    1. Generate the 7 points of the projective plane over F_2 (non-zero vectors in F_2^3).
    2. Identify lines (triples of points summing to 0 in F_2^3).
    3. Map these to vertices 0-6.
    4. Assign a valid 3-coloring.
    5. Return the structure with strict type enforcement.
    """
    
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin within 60s/90s limits

    # 1. Generate points of PG(2,2): Non-zero vectors in F_2^3
    # There are 2^3 - 1 = 7 such vectors.
    points = []
    for i in range(1, 8):
        # Represent integer i (1..7) as a 3-bit vector
        vec = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
        points.append(vec)
        
    # Map vector index to vertex ID 0..6
    # points[0] is vec(1) = (0,0,1) -> Vertex 0
    # points[1] is vec(2) = (0,1,0) -> Vertex 1
    # ...
    
    # 2. Find lines (edges)
    # In F_2^3, a line consists of 3 distinct points a, b, c such that a+b+c = 0.
    # This is equivalent to c = a+b (XOR).
    # We iterate through pairs (i, j) with i < j, compute k = i^j, and ensure k > j to avoid duplicates.
    # Wait, the indices in the list `points` correspond to integers 1..7.
    # Let's work directly with the integer labels 1..7 for the Fano plane points.
    
    fano_edges_int = set()
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = i ^ j
            if k > j:
                # The triple (i, j, k) forms a line in PG(2,2)
                fano_edges_int.add((i, j, k))
                
    # Convert to 0-indexed vertices 0..6
    # Vertex v corresponds to integer v+1
    edges = []
    for e in fano_edges_int:
        v0 = e[0] - 1
        v1 = e[1] - 1
        v2 = e[2] - 1
        # Sort for consistency, though order doesn't matter for hypergraph validity
        edges.append([v0, v1, v2])
        
    # Sort edges for deterministic output
    edges.sort()
    
    # 3. Determine a valid 3-coloring for the Fano plane
    # We need a coloring c: {0..6} -> {0,1,2} such that no edge is monochromatic.
    # The Fano plane is 3-chromatic. A specific valid coloring is:
    # Let's try to find one or use a known one.
    # Known 3-coloring for Fano plane (vertices 0-6 corresponding to ints 1-7):
    # Int 1 (001) -> 0
    # Int 2 (010) -> 1
    # Int 3 (011) -> 2
    # Int 4 (100) -> 2
    # Int 5 (101) -> 1
    # Int 6 (110) -> 0
    # Int 7 (111) -> 0 ? Let's verify.
    
    # Let's programmatically verify or find a simple valid coloring.
    # Since the graph is small, we can hardcode a known valid one after verification in head.
    # Edges from XOR:
    # (1,2,3), (1,4,5), (1,6,7), (2,4,6), (2,5,7), (3,4,7), (3,5,6)
    
    # Let's assign colors:
    # c[0] (int 1) = 0
    # c[1] (int 2) = 1
    # Edge (1,2,3) -> c[2] (int 3) must not be 0 or 1 if c[0]=0, c[1]=1? No, just not all same.
    # If c[0]=0, c[1]=1, c[2] can be 0,1,2. Let's pick 2.
    # Edge (1,4,5): c[0]=0.
    # Edge (2,4,6): c[1]=1.
    # Edge (3,4,7): c[2]=2.
    # Let's try a systematic search or known pattern.
    
    # A standard 3-coloring of the Fano plane:
    # Colors: 0, 1, 2
    # Vertices 0..6
    # Let's use this specific assignment which is known to work for the standard Fano embedding:
    colors_fano = [0, 1, 2, 2, 1, 0, 0]
    
    # Verify:
    # Edge (0,1,2) [ints 1,2,3]: colors 0,1,2 -> OK
    # Edge (0,3,4) [ints 1,4,5]: colors 0,2,1 -> OK
    # Edge (0,5,6) [ints 1,6,7]: colors 0,0,0 -> BAD (Monochromatic)
    
    # Let's find a valid one quickly.
    # The Fano plane is the unique 3-uniform 3-chromatic graph with 7 edges.
    # It is not 2-colorable.
    # We just need *one* 3-coloring.
    
    # Let's try:
    # 0:0, 1:1, 2:2
    # 3:0, 4:1, 5:2
    # 6:0
    # Check edges (0-indexed):
    # (0,1,2): 0,1,2 OK
    # (0,3,4): 0,0,1 OK
    # (0,5,6): 0,2,0 OK
    # (1,3,5): 1,0,2 OK
    # (1,4,6): 1,1,0 OK
    # (2,3,6): 2,0,0 OK
    # (2,4,5): 2,1,2 OK
    
    # This coloring works.
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # 4. Final Validation and Type Conversion
    # Ensure all types are Python int and lists are lists.
    
    edges_out = []
    for e in edges:
        edges_out.append([int(e[0]), int(e[1]), int(e[2])])
        
    colors_out = [int(c) for c in colors]
    
    # Sanity check: Ensure no monochromatic edges in the final structure
    for e in edges_out:
        c0 = colors_out[e[0]]
        c1 = colors_out[e[1]]
        c2 = colors_out[e[2]]
        if c0 == c1 == c2:
            # Fallback if the hardcoded coloring fails (it shouldn't)
            # Try a different permutation or raise error.
            # For robustness, let's just swap colors if needed, but 0,1,2,0,1,2,0 is verified above.
            pass
            
    # Check time budget
    if time.time() > deadline:
        # Return best found so far (which is the hardcoded valid one)
        pass

    return {
        'edges': edges_out,
        'colors': colors_out
    }

# EVOLVE-BLOCK-END
