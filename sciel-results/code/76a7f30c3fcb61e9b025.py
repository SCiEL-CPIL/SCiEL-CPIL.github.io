def construct():
    # Andrásfai graph A(9)
    # Triangle-free, independence number 9
    # n = 3*9 - 2 = 26
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
