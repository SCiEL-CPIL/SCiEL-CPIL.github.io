"""Sparse pancyclic graphs on 40 vertices: Hamiltonian cycle + minimal chords."""
import time

N = 40


def construct():
    t0 = time.time()
    DEADLINE = t0 + 75.0  # 75s out of 90s budget

    # Strategy: Hamiltonian cycle (40 edges) + 18 chords = 58 edges total.
    # Chords: (0,d) for d in 2..18 (17 chords) + (2,21) (1 chord).
    #
    # Cycle assignments:
    #   k=3..19:  forward path 0,1,...,k-1 + chord (0,k-1). Gap k-1 in 2..18.
    #   k=20:     forward path 2,3,...,21 + chord (2,21). 20 vertices.
    #   k=21:     composite: 0 ->(0,2) 2 ->(2,21) 21 ->22 ->...-> 39 ->0.
    #             Vertices: 0,2,21,22,...,39 = 21 distinct vertices.
    #   k=22:     backward path 2,1,0,39,38,...,21 + chord (2,21). 22 vertices.
    #   k=23..39: backward path 0,39,38,...,(41-k) + chord (0,41-k). Gap 41-k in 2..18.

    edges = set()

    # Hamiltonian cycle: (i, (i+1) % 40) for i = 0..39
    for i in range(N):
        a, b = i, (i + 1) % N
        edges.add((min(a, b), max(a, b)))

    # Chords: (0, d) for d in 2..18
    for d in range(2, 19):
        edges.add((0, d))

    # Chord: (2, 21)
    edges.add((2, 21))

    edges = sorted(edges)

    # Verify all edges in cycles exist (sanity check)
    edge_set = set(edges)

    cycles = []
    for k in range(3, N + 1):
        if k == N:
            # Hamiltonian cycle
            cyc = list(range(N))
        elif k <= 19:
            # Forward: 0, 1, 2, ..., k-1
            cyc = list(range(k))
        elif k == 20:
            # Forward: 2, 3, ..., 21
            cyc = list(range(2, 22))
        elif k == 21:
            # Composite: 0, 2, 21, 22, ..., 39
            cyc = [0, 2] + list(range(21, 40))
        elif k == 22:
            # Backward: 2, 1, 0, 39, 38, ..., 21
            cyc = [2, 1, 0] + list(range(39, 20, -1))
        else:
            # k in 23..39: backward: 0, 39, 38, ..., (41-k)
            dprime = 41 - k
            cyc = [0] + list(range(39, dprime - 1, -1))

        # Verify cycle validity
        assert len(cyc) == k, f"Cycle length {len(cyc)} != {k}"
        assert len(set(cyc)) == k, f"Duplicate vertices in cycle of length {k}"
        for i in range(k):
            u, v = cyc[i], cyc[(i + 1) % k]
            pair = (min(u, v), max(u, v))
            assert pair in edge_set, f"Edge {pair} missing for cycle of length {k}"

        cycles.append(cyc)

    # Time guard
    assert time.time() < DEADLINE, "Time limit exceeded"

    return {'edges': edges, 'cycles': cycles}
