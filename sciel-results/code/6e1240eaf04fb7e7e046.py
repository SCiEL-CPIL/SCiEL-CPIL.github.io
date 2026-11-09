"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 16
    # Vertices 0..7 and 8..15 form a complete bipartite graph K_{8,8}.
    # This is triangle-free, so every edge is a unique common-neighbor path.
    # Diameter is exactly 2. Removing any edge disconnects its endpoints
    # from having a length-2 path, increasing diameter to 3 (or disconnecting).
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
