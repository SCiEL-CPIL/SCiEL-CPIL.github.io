def construct():
    # Base graph H: Complement of 3x3 Rook Graph
    # Vertices 0..8 representing (i, j) where i = v // 3, j = v % 3
    n_h = 9
    edges_h = set()
    for u in range(n_h):
        i1, j1 = divmod(u, 3)
        for v in range(u + 1, n_h):
            i2, j2 = divmod(v, 3)
            if i1 != i2 and j1 != j2:
                edges_h.add((u, v))
    
    # Construct G: H[bar K_2]
    # Vertices 0..17
    # u maps to base u//2, offset u%2
    edges_g = []
    for u in range(18):
        b_u = u // 2
        for v in range(u + 1, 18):
            b_v = v // 2
            if b_u != b_v:
                if (b_u, b_v) in edges_h or (b_v, b_u) in edges_h:
                    edges_g.append([u, v])
    return edges_g
