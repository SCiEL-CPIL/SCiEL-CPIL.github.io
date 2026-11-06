def construct():
    # Vertices 0..15
    # Let U = {0, 1, 2}
    # Let S = {3, ..., 15}
    # H = all triples that intersect S
    # This is equivalent to all triples except those contained in U
    # Triples in U: {0, 1, 2} only.
    # So we return all triples except [0, 1, 2]
    
    triples = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                if not (a < 3 and b < 3 and c < 3):
                    triples.append([a, b, c])
    return triples
