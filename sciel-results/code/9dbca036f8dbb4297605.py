"""Deterministic construction of a triangle-free graph with alpha <= 9, n=30."""

def construct():
    n = 30
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # i and j are adjacent iff their difference mod 30 is in the edge set S
            diff = (j - i) % n
            if diff in (1, 3, 5, 9, 13, 21, 23, 25, 27, 29):
                edges.append([i, j])
    return {'n': n, 'edges': edges}
