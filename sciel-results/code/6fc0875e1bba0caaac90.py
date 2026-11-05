def construct():
    edges = []
    for i in range(10):
        for j in range(10):
            u = i
            v = 10 + j
            c = (i * 10 + j) % 20
            edges.append([u, v, c])
    return edges
