"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # 8 disjoint K4s plus one extra vertex (32) connected to exactly one vertex per K4.
    # This gives n=33, no C5, and alpha <= 8.
    n_cliques = 8
    clique_size = 4
    base_n = n_cliques * clique_size  # 32
    extra_vertex = base_n  # 32
    n = base_n + 1  # 33
    edges = []
    for b in range(n_cliques):
        base = b * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append([base + i, base + j])
    # Connect extra vertex to one vertex from each K4 (e.g., the first vertex of each clique)
    for b in range(n_cliques):
        base = b * clique_size
        edges.append([extra_vertex, base])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
