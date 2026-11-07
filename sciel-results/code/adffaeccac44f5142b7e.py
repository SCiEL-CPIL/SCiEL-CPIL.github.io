def construct():
    edges = []
    # K_7,8
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
    # One edge in A
    edges.append([0, 1])
    # One edge in B
    edges.append([7, 8])
    return edges
