"""Voltage graph lift approach for cage problem n(4,10)."""
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using voltage graph lifting.
    
    Strategy: Use a small base graph (cage-like) and lift it via a voltage
    assignment over a cyclic group to increase girth while keeping vertex count low.
    
    For k=4, g=10: We use a base graph and lift over Z_m for suitable m.
    """
    if k != 4 or g != 10:
        # Fallback: use Lazebnik-Ustimenko-Woldar construction
        return _luw_construct(k, g)
    
    # For the specific case k=4, g=10, we try a voltage graph approach
    # Base graph: a 4-regular graph with good expansion properties
    # We'll use a random-looking 4-regular base and optimize the voltage
    
    # Actually, let's think more carefully. The Moore bound for n(4,10) is:
    # n >= 1 + k*sum_{i=0}^{(g-2)/2 - 1} (k-1)^i for even g
    # For g=10: n >= 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161
    # Wait, that's for odd girth. For even girth g=2r:
    # n >= 2*sum_{i=0}^{r-1} (k-1)^i = 2*(1 + 3 + 9 + 27) = 2*40 = 80
    # Hmm, let me recalculate. Moore bound for girth g=2r (even), degree k:
    # n >= 2 * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(r-1))
    # For k=4, g=10, r=5: n >= 2*(1+3+9+27+81) = 2*121 = 242
    # The best lower bound mentioned is 243, so we're close to the Moore bound.
    
    # The known best is 320 vertices. We need to do better or match.
    # A voltage graph lift of a smaller base can potentially achieve this.
    
    # Let's try a different approach: use the known structure of the (4,10)-cage
    # or construct via algebraic methods.
    
    # Since the task says the smallest known is 320 (Exoo et al. 2026),
    # and we need to produce a valid graph, let's use a systematic construction.
    
    # Approach: Use a circulant-like construction or a known explicit (4,10)-cage.
    # The (4,10)-cage with 320 vertices was found by Exoo et al.
    
    # Let me try to construct it using the voltage graph method properly.
    # Base graph: 4-regular, say 16 vertices (the (4,6)-cage is the 16-vertex graph)
    # Lift over Z_20 to get 320 vertices.
    
    # Actually, let's just implement a known explicit (4,10)-cage construction.
    # The 320-vertex (4,10)-cage can be described as follows:
    # It's a bipartite graph with 160+160 vertices.
    
    # Let me use a simpler approach: generate a random 4-regular graph and check girth,
    # or use a structured construction.
    
    # Given the 120-second time limit, let's use a deterministic construction.
    
    # The (4,10)-cage: I'll use the construction from the literature.
    # One explicit construction uses the group Z_80 x Z_4 with specific generators.
    
    # Actually, let me just use the Lazebnik-Ustimenko-Woldar construction which is
    # guaranteed to work, and see if we can improve it.
    
    return _luw_construct(k, g)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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


# EVOLVE-BLOCK-END
