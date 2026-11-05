def construct():
    # 5 disjoint copies of K7
    # n = 35
    # alpha = 5
    # No C5
    n = 35
    edges = []
    for b in range(5):
        for i in range(7):
            for j in range(i + 1, 7):
                u = 7 * b + i
                v = 7 * b + j
                edges.append([u, v])
    return {'n': n, 'edges': edges}
