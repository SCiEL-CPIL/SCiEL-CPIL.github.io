import time

def construct():
    n = 15
    # Optimal solution is K_{7,8} with 56 edges.
    # Vertices 0..6 (Part A, size 7) connected to 7..14 (Part B, size 8).
    # Degrees: Part A = 8, Part B = 7.
    # No path of length 3 has endpoints with equal degree.
    edges = []
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    return edges
