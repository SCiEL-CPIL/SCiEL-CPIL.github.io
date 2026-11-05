"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a complete graph K_20 with a 1-factorization edge coloring.
    This ensures that every 4-cycle is rainbow.
    Max edges = 190.
    """
    n = 20
    edges = []
    
    # We use vertices 0 to 18 for the cyclic part, and 19 as the 'infinity' vertex.
    # The 1-factorization of K_{2n} for n=10 has 2n-1 = 19 factors.
    # Each factor is a perfect matching of size 10.
    
    for r in range(19):
        color = r
        # The edge connecting the fixed vertex 19 to vertex r
        edges.append([19, r, color])
        
        # The remaining 18 vertices (0..18 excluding r) are paired up.
        # Pair (r-i) and (r+i) modulo 19.
        for i in range(1, 10):
            u = (r - i) % 19
            v = (r + i) % 19
            # Normalize order for cleanliness, though not strictly required
            if u > v:
                u, v = v, u
            edges.append([u, v, color])
            
    return edges
# EVOLVE-BLOCK-END
