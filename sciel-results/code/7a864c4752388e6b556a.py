import time

N = 40

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0

    # Base: 40-cycle (40 edges)
    base_edges = set()
    for i in range(N):
        base_edges.add((i, (i + 1) % N))

    # Chords: (0, k) for k in 2..20 (19 chords)
    # This gives cycles of all lengths 3..40 with 59 total edges.
    # Let's verify the cycle construction:
    # For cycle of length L:
    #   If L == 40: the entire 40-cycle.
    #   If L < 40: we use vertex 0, then traverse the base cycle from 0 for L-1 steps,
    #     but we need a chord to close the cycle.
    #     Actually, with chords (0, k) for k=2..20:
    #     To make a cycle of length L (3 <= L <= 21):
    #       Use vertices 0, 1, 2, ..., L-1. Edges: (0,1), (1,2), ..., (L-2, L-1) from base cycle.
    #       Need edge (L-1, 0). That's a chord if L-1 is in 2..20, i.e., L in 3..21.
    #       For L=21: chord (0,20). OK.
    #     For L in 22..39:
    #       Use vertices 0, 40-L+1, 40-L+2, ..., 39. That is 0 and the last (L-1) vertices.
    #       Wait, let me think again.
    #       Vertices: 0, k, k+1, ..., 39 where k = 41-L.
    #       Number of vertices: 1 + (39 - k) + 1 = 41 - k = L.
    #       Edges: (0, k) is a chord (since k = 41-L, and for L=22..39, k=19..2, so k in 2..19).
    #       Then (k, k+1), ..., (38, 39) from base cycle.
    #       Then (39, 0) from base cycle.
    #       So the cycle is 0 -> k -> k+1 -> ... -> 39 -> 0.
    #       That's L vertices.

    chord_edges = set()
    for k in range(2, 21):  # 2 to 20
        chord_edges.add((0, k))

    all_edges = list(base_edges | chord_edges)
    
    # Build cycles
    cycles = []
    for L in range(3, N + 1):
        if L == 40:
            cycles.append(list(range(40)))
        elif L <= 21:
            # Cycle: 0, 1, 2, ..., L-1
            cyc = list(range(L))
            cycles.append(cyc)
        else:
            # L in 22..39
            k = 41 - L  # k is in 2..19
            # Cycle: 0, k, k+1, ..., 39
            cyc = [0] + list(range(k, 40))
            cycles.append(cyc)

    return {'edges': all_edges, 'cycles': cycles}

# EVOLVE-BLOCK-END
