"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 15}

def construct():
    n = P['n']
    # Split vertices into two parts as evenly as possible to maximize edges
    # while maintaining the bipartite property which prevents 3-paths
    # between vertices of the same degree class.
    # For n=15, split 7 and 8.
    mid = n // 2
    part_a = list(range(mid))
    part_b = list(range(mid, n))
    
    edges = []
    for u in part_a:
        for v in part_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
