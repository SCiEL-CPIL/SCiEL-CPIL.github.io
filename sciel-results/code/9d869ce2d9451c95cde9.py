def construct():
    # 8 disjoint copies of K4
    # n = 32
    # alpha = 8
    # No C5
    n = 32
    edges = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                u = 4 * b + i
                v = 4 * b + j
                edges.append([u, v])
    return {'n': n, 'edges': edges}
