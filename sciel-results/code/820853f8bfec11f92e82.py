def construct():
    # 7 disjoint copies of K5
    # n = 35
    # alpha = 7
    # C5-free
    n = 35
    edges = []
    for b in range(7):
        base = b * 5
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
