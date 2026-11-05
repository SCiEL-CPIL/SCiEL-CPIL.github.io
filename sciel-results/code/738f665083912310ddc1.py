N = 40

def construct():
    # Fan graph: base cycle 0-1-...-39-0 plus chords (0,k) for k=2..20
    # Total edges: 40 + 19 = 59, value = 19
    edge_set = set()
    for i in range(N):
        edge_set.add((i, (i + 1) % N))
    for k in range(2, 21):
        edge_set.add((0, k))

    edges = [tuple(sorted(e)) for e in sorted(edge_set)]

    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cyc = list(range(N))
        elif L == 21:
            cyc = list(range(21))
        elif L <= 20:
            cyc = list(range(L))
        else:
            k = 41 - L
            cyc = [0] + list(range(k, N))
        cycles.append(cyc)

    return {'edges': edges, 'cycles': cycles}
