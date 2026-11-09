"""Rewritten program: complement of the 5x6 rook graph (line graph of K_{5,6})."""

def construct():
    n = 30
    edges = []
    for i in range(5):
        for j in range(6):
            for i2 in range(5):
                for j2 in range(6):
                    if i == i2 and j == j2:
                        continue
                    u = 6 * i + j
                    v = 6 * i2 + j2
                    if u < v:
                        edges.append([u, v])
    return {'n': n, 'edges': edges}
