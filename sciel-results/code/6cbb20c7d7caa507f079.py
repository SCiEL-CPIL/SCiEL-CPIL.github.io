import time

def construct():
    """
    Sparse pancyclic graph on 40 vertices.
    Returns a dict with 'edges' (list of undirected edge pairs) and 'cycles'
    (list of vertex-list cycles, one for each length 3..40 in that order).
    """
    N = 40
    edges = set()

    # Add the Hamiltonian cycle edges: (0,1), (1,2), ..., (38,39), (39,0)
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Add 16 chords from vertex 0 to vertices 2..17
    for k in range(2, 18):
        edges.add((0, k))

    # Add 1 chord from vertex 1 to vertex 22
    edges.add((1, 22))

    # Total edges: 40 + 16 + 1 = 57

    # Build the edge list with consistent ordering (min, max)
    edge_list = []
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_list.append((u, v))

    # Build cycles of every length 3..40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full Hamiltonian cycle: 0, 1, 2, ..., 39
            cyc = list(range(N))
        elif 3 <= L <= 18:
            # Cycle: 0 -> 1 -> 2 -> ... -> L-1 -> 0
            # Edges used: (0,1), (1,2), ..., (L-2,L-1), (L-1,0)
            # The chord (0, L-1) exists since 2 <= L-1 <= 17
            cyc = list(range(L))
        elif 19 <= L <= 23:
            # Cycle using chord (1, 22): 0 -> 1 -> 22 -> 21 -> ... -> k -> 0
            # where k = 22 - L + 3 = 25 - L
            # Vertices: 0, 1, 22, 21, ..., k (total L vertices)
            # Edges used: (0,1), (1,22), (22,21), ..., (k+1,k), (k,0)
            # The chord (k,0) exists since k = 25-L, and for L=19..23, k=6..2
            k = 25 - L
            cyc = [0, 1] + list(range(22, k - 1, -1))
        else:
            # Cycle: 0 -> 39 -> 38 -> ... -> k -> 0 where k = 41 - L
            # Vertices: 0, 39, 38, ..., k (total L vertices)
            # Edges used: (0,39), (39,38), ..., (k+1,k), (k,0)
            # The chord (0, k) exists since 2 <= k <= 17
            k = 41 - L
            cyc = [0] + list(range(39, k - 1, -1))
        cycles.append(cyc)

    return {'edges': edge_list, 'cycles': cycles}
