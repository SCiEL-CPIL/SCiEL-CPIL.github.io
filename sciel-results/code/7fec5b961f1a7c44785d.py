import time

N = 40

def construct():
    # Deadline guard: 90s limit, use 75s for safety
    deadline = time.time() + 75

    # 1. Base Cycle Edges (Hamiltonian cycle 0-1-...-39-0)
    edges = set()
    for i in range(N):
        j = (i + 1) % N
        edges.add((min(i, j), max(i, j)))

    # 2. Add Chords
    # Fan construction from vertex 0.
    # Chords (0, k) for k in 2..20.
    # This creates cycles of all lengths 3..39.
    # k=2 -> L=3, 39
    # k=3 -> L=4, 38
    # ...
    # k=19 -> L=20, 22
    # k=20 -> L=21
    
    chords_k = range(2, N // 2 + 1)

    for k in chords_k:
        e = (0, k)
        edges.add(e)

    edge_list = [tuple(sorted(e)) for e in edges]

    # 3. Construct Cycles
    cycles = []

    for L in range(3, N + 1):
        if L == N:
            cycle_verts = list(range(N))
        else:
            # Try to use chord (0, L-1)
            k1 = L - 1
            k2 = N + 1 - L

            # We added chords for k in 2..20.
            if 2 <= k1 <= N // 2:
                # Cycle: 0 -> 1 -> ... -> k1 -> 0
                # Vertices: 0, 1, ..., k1
                # Length: k1 + 1 = L.
                cycle_verts = list(range(k1 + 1))
            else:
                # k1 > 20. Then k2 < 20.
                # k2 = 41 - L.
                # Since L <= 39, k2 >= 2.
                # So k2 is in 2..19.
                # Use chord (0, k2).
                # Cycle: 0 -> k2 -> k2+1 -> ... -> N-1 -> 0
                # Vertices: 0, k2, k2+1, ..., N-1
                # Length: 1 + (N-1 - k2) + 1 = N - k2 + 1 = 41 - k2 = L.
                cycle_verts = [0] + list(range(k2, N))

        cycles.append(cycle_verts)

    return {'edges': edge_list, 'cycles': cycles}
