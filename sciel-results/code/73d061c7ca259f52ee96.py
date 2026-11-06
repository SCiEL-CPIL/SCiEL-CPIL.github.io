N = 40

def construct():
    # Build a sparse pancyclic graph on 40 vertices.
    # Base cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))

    # Add chords from vertex 0 to vertices 2..20.
    # Each chord (0, j) creates two cycles of lengths j+1 and N+1-j.
    # For j in 2..20, we get lengths 3..21 and 39..21, covering 3..40.
    for j in range(2, 21):
        edges.add((0, j))

    # Normalize edges to (min, max) tuples
    edge_list = []
    for a, b in edges:
        if a > b:
            a, b = b, a
        edge_list.append((a, b))

    # Generate one cycle of each length L = 3, 4, ..., 40
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # The base Hamiltonian cycle
            cycles.append(list(range(N)))
        elif L <= 21:
            # Use chord (0, L-1): cycle 0-1-2-...-(L-1)-0
            cycles.append(list(range(L)))
        else:
            # Use chord (0, j) where j = N + 1 - L
            # Cycle: 0-j-(j+1)-...-(N-1)-0
            j = N + 1 - L
            cycles.append([0] + list(range(j, N)))

    return {'edges': edge_list, 'cycles': cycles}
