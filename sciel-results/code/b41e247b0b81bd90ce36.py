def construct():
    n = 16
    left = [i for i in range(8)]
    right = [i for i in range(8, 16)]
    edges = [[u, v] for u in left for v in right]
    return edges
