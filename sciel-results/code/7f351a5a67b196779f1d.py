import time

N = 40

def construct():
    # Base cycle edges: 0-1-2-...-39-0 (40 edges)
    edges = []
    edge_set = set()

    for i in range(N):
        u = i
        v = (i + 1) % N
        e = (min(u, v), max(u, v))
        edges.append(e)
        edge_set.add(e)

    # Add 19 chords from vertex 0 to vertices 2..20
    # Chord (0,d) certifies cycle length d+1 (forward) and 41-d (backward)
    # Together these cover all lengths 3..39
    for d in range(2, 21):
        e = (0, d)
        edges.append(e)
        edge_set.add(e)

    # Generate cycles for lengths 3..40
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycle = list(range(N))
        else:
            # Try forward: 0, 1, 2, ..., k-1 (needs edge (0, k-1))
            d1 = k - 1
            if d1 >= 2 and d1 <= 20:
                cycle = list(range(d1 + 1))
            else:
                # Try backward: 0, N-1, N-2, ..., N-k+1 (needs edge (0, N-k+1))
                d2 = N - (k - 1)
                cycle = [0] + list(range(N - 1, d2 - 1, -1))
        cycles.append(cycle)

    return {'edges': edges, 'cycles': cycles}
