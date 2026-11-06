"""Pentagon-free graph with independence number < 9 for R(C5,K9) lower bound."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with n=32 vertices that is C5-free and has independence number 8.
    Construction: 8 disjoint copies of K4.
    - C5-free: K4 contains no cycles of length 5.
    - Independence number: alpha(K4) = 1, so alpha(8*K4) = 8 < 9.
    """
    n = 32
    clique_size = 4
    num_cliques = 8
    
    edges = []
    for c in range(num_cliques):
        start = c * clique_size
        # Generate all edges within this clique
        for i in range(start, start + clique_size):
            for j in range(i + 1, start + clique_size):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
