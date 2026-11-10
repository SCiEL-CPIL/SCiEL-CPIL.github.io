"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 16
    # The graph must be triangle-free to satisfy the edge-criticality condition for diameter.
    # The maximum triangle-free graph on 16 vertices is K_8,8 with 64 edges.
    # K_8,8 has diameter 2 and removing any edge (u,v) makes dist(u,v) = 3 (via a path of length 3), 
    # thus increasing the diameter from 2 to 3.
    
    partition_a = list(range(8))
    partition_b = list(range(8, 16))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
