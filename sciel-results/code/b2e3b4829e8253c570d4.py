def construct():
    """
    Pentagon-free graph without independent 9-sets.
    Returns n=33: 8 disjoint K4 components plus one extra vertex connected to all
    vertices of one K4. This preserves C5-freeness and keeps alpha <= 8.
    """
    n_cliques = 8
    clique_size = 4
    n = n_cliques * clique_size + 1  # 33
    extra_vertex = n - 1  # 32

    edges = []

    # Build 8 disjoint K4 cliques
    for b in range(n_cliques):
        base = b * clique_size
        for i in range(clique_size):
            for j in range(i + 1, clique_size):
                edges.append([base + i, base + j])

    # Connect extra vertex to all 4 vertices of the first K4 (clique 0)
    for i in range(clique_size):
        edges.append([extra_vertex, i])

    return {'n': n, 'edges': edges}
