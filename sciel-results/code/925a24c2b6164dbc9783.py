"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct the Clebsch graph on 16 vertices.
    The Clebsch graph is a triangle-free graph with chromatic number 4.
    It has 16 vertices and 40 edges, which is the maximum number of edges
    for a triangle-free 4-chromatic graph on 16 vertices.
    
    Construction:
    Vertices are integers 0..15 (4-bit binary strings).
    Two vertices u and v are adjacent if their Hamming distance is 3 or 4.
    This is equivalent to the complement of the graph formed by edges of 
    length 1 and 2 in the 4-dimensional hypercube Q4.
    """
    edges = []
    n = 16
    for u in range(n):
        for v in range(u + 1, n):
            # Calculate Hamming distance
            xor_val = u ^ v
            # Count bits
            dist = bin(xor_val).count('1')
            if dist == 3 or dist == 4:
                edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
