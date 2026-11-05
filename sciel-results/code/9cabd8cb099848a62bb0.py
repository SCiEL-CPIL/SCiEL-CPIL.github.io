import time

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 85

    # Hamiltonian cycle: 0-1-2-...-39-0
    edges = set()
    for i in range(N):
        edges.add((min(i, (i + 1) % N), max(i, (i + 1) % N)))

    adj = {i: set() for i in range(N)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Cycle certificates
    cycle_cert = {N: list(range(N))}

    def get_chord_cycles(i, j):
        """Return two cycles formed by adding chord (i,j) to Hamiltonian cycle."""
        if i == j:
            return None, None
        if i < j:
            c1 = list(range(i, j + 1))
            c2 = list(range(j, N)) + list(range(0, i + 1))
        else:
            c1 = list(range(i, N)) + list(range(0, j + 1))
            c2 = list(range(j, i + 1))
        return c1, c2

    missing = set(range(3, N))

    while missing and time.time() < deadline:
        best_chord = None
        best_cover = set()
        best_cycles = {}

        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in edges:
                    continue
                c1, c2 = get_chord_cycles(i, j)
                l1, l2 = len(c1), len(c2)
                cover = set()
                if l1 in missing:
                    cover.add(l1)
                if l2 in missing:
                    cover.add(l2)
                if len(cover) > len(best_cover):
                    best_cover = cover
                    best_chord = (i, j)
                    best_cycles = {}
                    if l1 in missing:
                        best_cycles[l1] = c1
                    if l2 in missing:
                        best_cycles[l2] = c2
                    if len(cover) >= 2:
                        break
            if len(best_cover) >= 2:
                break

        if best_chord is None:
            break

        i, j = best_chord
        edges.add((i, j))
        adj[i].add(j)
        adj[j].add(i)

        for l, c in best_cycles.items():
            if l in missing:
                cycle_cert[l] = c
                missing.discard(l)

    cycles = []
    for k in range(3, N + 1):
        if k in cycle_cert:
            cycles.append(cycle_cert[k])
        else:
            cycles.append(list(range(k)))

    return {'edges': list(edges), 'cycles': cycles}
