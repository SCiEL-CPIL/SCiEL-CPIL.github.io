"""Deterministic structured construction for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 20
    edges = []
    # 1-factorization of K20 using cyclic shifts modulo 19
    for shift in range(19):
        # Pair the infinity vertex (19) with vertex (19 - shift) % 19
        a = 19
        b = (19 - shift) % 19
        edges.append([a, b, shift])
        
        # Pair the remaining 18 vertices (0..18)
        for i in range(9):
            u = (shift + i) % 19
            v = (shift - i - 1) % 19
            edges.append([u, v, shift])
            
    return edges
# EVOLVE-BLOCK-END
