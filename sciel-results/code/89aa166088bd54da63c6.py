N = 40

def construct():
    # Build a sparse pancyclic graph on 40 vertices.
    # Strategy: use a base cycle plus a set of chords from vertex 0 to create
    # cycles of every length 3..40 with minimal extra edges.
    #
    # Edges:
    #   - Cycle edges: (i, (i+1) % N) for i in 0..39
    #   - Chords: (0, k) for k in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    #
    # Total edges = 40 (cycle) + 19 (chords) = 59
    # Score = 59 - 40 = 19

    edge_set = set()

    # Add cycle edges
    for i in range(N):
        a, b = i, (i + 1) % N
        edge_set.add((min(a, b), max(a, b)))

    # Add chords from vertex 0 to vertices 2..20
    for k in range(2, 21):
        a, b = 0, k
        edge_set.add((min(a, b), max(a, b)))

    # Sort edges for deterministic output
    edges = sorted(edge_set)

    # Construct cycles of every length L from 3 to 40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Full cycle: 0, 1, 2, ..., 39
            cyc = list(range(N))
        elif L <= 21:
            # Cycle 0, 1, 2, ..., L-1 uses cycle edges (0-1, 1-2, ..., (L-2)-(L-1))
            # and chord (L-1, 0) which exists since L-1 <= 20
            cyc = list(range(L))
        else:
            # L in 22..39: use cycle 0, k, k+1, ..., 39 where k = 41 - L
            # This uses chord (0, k), cycle edges (k, k+1), ..., (38, 39), (39, 0)
            # Length: 1 (vertex 0) + (N - k) (vertices k..39) = 1 + 40 - (41-L) = L
            k = 41 - L
            cyc = [0] + list(range(k, N))
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
