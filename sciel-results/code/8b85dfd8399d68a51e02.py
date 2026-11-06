import time

def construct():
    """
    Constructs a triangle-free graph with independence number < 10.
    Uses the cyclic Andrássai graph construction on Z_n where n=27.
    Vertices are 0..n-1.
    Edge (i, j) exists if (j - i) mod n is congruent to 1 mod 3.
    For n=27, this creates a graph with 3 independent sets of size 9 (based on mod 3 classes),
    and complete bipartite connections between them, resulting in alpha(G) = 9.
    This is triangle-free.
    """
    n = 27
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}
