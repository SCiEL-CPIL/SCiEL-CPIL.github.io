"""Constructs a sparse pancyclic graph on 40 vertices.

The construction is a C40 cycle with 19 chords incident to vertex 0
(0,2),(0,3),...,(0,20). This is the classic "elementary fan" seed: the
cycle provides length-40; each chord (0,d) for d=2..20 closes a cycle
of length d+1 on the forward arc 0->1->...->d, and by symmetry the
complementary arc 0->39->...->(N-d) closes a cycle of length (N-d)+1.
Together these cover every length 3..39, so total edges = 40+19 = 59,
minimizing (edges - 40) = 19.
"""

N = 40
_HALF = N // 2  # 20


def _cycle_for_length(k: int) -> list:
    """Return a simple cycle (list of distinct vertices) of length k on
    vertices 0..N-1 using only the base cycle edges and chords (0,d)
    for d in 2..N-2."""
    if k == N:
        # The Hamiltonian cycle around the base ring.
        return list(range(N))
    d = k - 1  # circular distance between endpoints of the closing chord
    if d <= _HALF:
        # Forward arc: 0,1,2,...,d ; closing edge (0,d) is a chord.
        return list(range(d + 1))
    else:
        # Backward arc: 0,N-1,N-2,...,N-d ; closing edge (0,N-d) is a chord.
        # Length of this arc is d+1 = k.
        tail = list(range(N - 1, N - d - 1, -1))
        return [0] + tail


def construct():
    # Base cycle edges (undirected, written in canonical order).
    edges = []
    edge_set = set()

    def _add(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        if (a, b) not in edge_set:
            edge_set.add((a, b))
            edges.append((a, b))

    for i in range(N):
        _add(i, (i + 1) % N)

    # Fan chords from vertex 0 to vertices 2..20.
    for i in range(2, _HALF + 1):
        _add(0, i)

    # Certificates: one cycle for every length 3..40, in order.
    cycles = [_cycle_for_length(k) for k in range(3, N + 1)]

    return {'edges': edges, 'cycles': cycles}
