"""Fresh design: hub + path pancyclic certificate on 40 vertices."""
N = 40

# EVOLVE-BLOCK-START
def construct():
    # Spoke edges from hub 0 to every other vertex.
    edges = [(0, i) for i in range(1, N)]
    # Path edges 1-2-...-38 to support cycles of length 3..39.
    edges += [(i, i + 1) for i in range(1, N - 2)]
    # Extra edge to close the length-40 cycle 0,1,...,39,38.
    edges.append((N - 2, N - 1))

    # Certificate: one distinct-vertex cycle of every length 3..40, in order.
    cycles = []
    for k in range(3, N):  # 3..39
        cycles.append([0] + list(range(1, k)))
    cycles.append([0] + list(range(1, N)) + [N - 2])  # length 40

    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END
