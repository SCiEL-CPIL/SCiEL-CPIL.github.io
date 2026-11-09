"""Dense odd-order graphs avoiding equal-degree path endpoints."""

def construct():
    n = 15
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (i == 0 and j > 0):
                edges.append([i, j])
    return edges
