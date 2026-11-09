import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s kill

    # The strongest idea from the inspirations: use the LUW construction with
    # the *minimal* d that guarantees girth >= g, which yields ~13k vertices
    # for (3,16) instead of ~350k.  Then, use the local-search machinery from
    # the current program to try to prune the graph down further.
    #
    # Structural change: instead of trying random 3-regular graphs (which will
    # almost certainly have short cycles), we start from a *known* high-girth
    # graph (LUW) and attempt to remove vertices while preserving girth.

    # Build LUW graph with minimal d
    luw_n, luw_edges = _luw_construct(k, g)

    # If the LUW graph already meets the girth requirement (it should by
    # construction), try to reduce its size via local search / vertex removal.
    #
    # Strategy: repeatedly try to remove a vertex and its neighbours, then
    # "re-wire" to maintain 3-regularity.  This is a 2-switch / edge-swap
    # operation on the complement.
    #
    # A simpler and more effective approach for bipartite graphs: use the
    # fact that LUW graphs are bipartite.  We can try to find and remove
    # "redundant" cycles, or use a simulated-annealing style local search
    # to find a smaller 3-regular bipartite graph with girth >= g.

    # Given the time budget, the most promising structural improvement is:
    # 1. Build the minimal LUW graph (fast, deterministic).
    # 2. Try a limited number of random 2-switches to see if we can find a
    #    smaller connected component or reduce the total vertex count.
    # 3. If no improvement, return the LUW graph.

    # Actually, the LUW graph is already connected and 3-regular.  The only
    # way to get fewer vertices is to find a *different* 3-regular graph
    # with girth >= g that is smaller.  This is the hard part.
    #
    # A practical approach within the time budget:
    # - Use the LUW graph as a starting point.
    # - Try to "contract" it by finding pairs of vertices that can be merged
    #   while preserving 3-regularity and girth.  This is complex.
    #
    # Given the constraints, the best we can do is:
    # - Return the minimal LUW graph (score ~3 based on the findings).
    # - Spend remaining time trying a few random 2-switches to see if we can
    #   accidentally find a smaller valid graph (very unlikely).

    # Let's try a different structural approach: use a *different* family of
    # high-girth graphs.  For (3,16), the known best is 936.  The LUW graph
    # with minimal d gives ~13k.  There's a big gap.
    #
    # A better structural idea: use a *product* or *lexicographic* construction
    # of smaller high-girth graphs.  For example, if we have a (3,8)-graph
    # with n1 vertices and a (3,8)-graph with n2 vertices, we can sometimes
    # construct a (3,16)-graph with n1*n2 vertices.  But this doesn't help
    # directly.
    #
    # Given the time constraints, I'll focus on making the LUW construction
    # as efficient as possible and returning it.  The local search from the
    # current program is unlikely to help within the time budget for such a
    # large graph.

    return luw_n, luw_edges


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    
    Returns a q-regular bipartite graph with girth at least d + 5.
    We choose the minimal d such that d + 5 >= g.
    """
    q = k
    # Minimal d such that d + 5 >= g, i.e., d >= g - 5
    d = g - 5
    if d < 3:
        d = 3
    # Ensure d is odd for the standard LUW construction
    if d % 2 == 0:
        d += 1  # This might overshoot, but it's safe

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
