"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the Clebsch graph on 16 vertices (indices 0-15)
    # Clebsch graph is the complement of the halved 5-cube.
    # Vertices are even-parity 5-bit strings.
    # Edges exist if Hamming distance is 4.
    
    vertices = []
    for i in range(32):
        # Check if i has even parity (even number of 1s)
        # i is 0..31, so 5 bits
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
            
    # vertices should have 16 elements
    # Map original value to index 0..15
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    
    # Add Clebsch edges
    for i in range(16):
        for j in range(i + 1, 16):
            v_i = vertices[i]
            v_j = vertices[j]
            # Hamming distance
            diff = v_i ^ v_j
            dist = bin(diff).count('1')
            if dist == 4:
                edges.append([i, j])
                
    # Add K4 on vertices 16, 17, 18, 19
    for i in range(16, 20):
        for j in range(i + 1, 20):
            edges.append([i, j])
            
    return edges
# EVOLVE-BLOCK-END
