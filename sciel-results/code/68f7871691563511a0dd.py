def construct():
    n = 16
    return [(i + j) % 15 for i in range(n) for j in range(i + 1, n)]
