"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 40
    set_a = list(range(20))
    set_b = list(range(20, 40))
    
    edges = []
    
    # Add all bipartite edges K_{20, 20}
    for u in set_a:
        for v in set_b:
            edges.append([u, v])
            
    # Add exactly one intra-partition edge
    # Choose (0, 1) in set_a
    edges.append([0, 1])
    
    return edges
# EVOLVE-BLOCK-END
