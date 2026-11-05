"""
Sparse pancyclic graph on 40 vertices.

Strategy:
  - Base: a single 40-cycle on vertices 0..39  (40 edges)
  - Chords: (0, 2), (0, 3), ..., (0, 20)  (19 edges)
  Total edges = 59, so the value (edges - 40) = 19.

Cycle certificates:
  For each L in 3..40:
    L = 40  -> the base Hamiltonian cycle [0, 1, 2, ..., 39]
    L in 3..21  -> chord (0, L-1): cycle 0, 1, ..., L-1, 0  (length L)
    L in 22..39 -> chord (0, 40-L): cycle 0, 40-L, 40-L+1, ..., 39, 0 (length L)
"""

import time


def construct():
    start = time.time()

    N = 40

    # Build edge set
    edge_set = set()

    # Base Hamiltonian cycle
    for i in range(N):
        j = (i + 1) % N
        edge_set.add((min(i, j), max(i, j)))

    # Chords from vertex 0 to vertices 2..20
    for k in range(2, N // 2 + 1):  # k = 2..20
        edge_set.add((0, k))

    # Deterministic edge list
    edge_list = sorted(edge_set)

    # Build cycle certificates
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            # Hamiltonian cycle
            cycles.append(list(range(N)))
        elif L <= N // 2 + 1:
            # Use chord (0, L-1): cycle 0, 1, 2, ..., L-1
            cycles.append(list(range(L)))  # [0, 1, 2, ..., L-1]
        else:
            # L in 22..39: use chord (0, N-L)
            # Cycle: 0, N-L, N-L+1, ..., N-1
            k = N - L  # this is in range 2..18
            cycles.append([0] + list(range(k, N)))

    return {'edges': edge_list, 'cycles': cycles}
