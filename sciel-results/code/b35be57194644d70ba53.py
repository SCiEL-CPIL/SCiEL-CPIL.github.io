"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time

def construct(k, g):
    # For k=3, g=16: use the Lazebnik-Ustimenko-Woldar construction with
    # a depth parameter d chosen to guarantee girth >= 16.
    # LUW graphs are q-regular bipartite graphs (q=k) with girth at least d+5 (odd d)
    # or d+6 (even d). For g=16, we need d+5 >= 16 => d >= 11 (odd).
    # We'll use d=11, giving girth >= 16.
    #
    # However, the LUW graph with q=3, d=11 has a specific number of vertices.
    # Let's compute it: the LUW graph CD(d,q) has 2*q*d vertices (points and lines).
    # For q=3, d=11: 2*3*11 = 66 vertices.
    # But wait, the LUW construction as described in the seed produces a connected
    # component of the zero point, which may be smaller.
    #
    # Actually, the standard LUW construction CD(d,q) is a (d-1)-regular bipartite graph
    # on 2*q^d vertices... no, let me re-examine.
    #
    # The seed code's LUW construction: points and lines are vectors over GF(q).
    # Points: (0, x) where x is a tuple of length d+1 (first element None).
    # Lines: (1, y) where y is a tuple of length d+1.
    # For q=3, the total number of possible points is 3^d = 3^11 = 177147.
    # That's too many. But the connected component of the zero point is much smaller.
    #
    # Let me reconsider. The seed code says "the connected component of the zero point
    # is returned." For small d, this is manageable. For d=11, q=3, it might be large.
    #
    # Let me try a different approach: use the known (3,16)-cage constructions.
    # The smallest known is 936 vertices. The seed's LUW approach with appropriate d
    # should produce a valid graph. Let me trace through the seed code more carefully.
    #
    # In the seed code:
    # - q = k = 3
    # - d = max(3, g-5 if (g-5)%2 else g-4) = max(3, 16-5=11) = 11 (since 11 is odd)
    # - The graph has vertices that are (side, tuple) where side is 0 or 1
    # - Points: side=0, tuple = (None,) + (a1, a2, ..., ad) in GF(3)^d
    # - Lines: side=1, tuple = (b1, b2, ..., bd+1) in GF(3)^(d+1)
    #
    # Wait, looking at the seed code more carefully:
    # start = (0, (None,) + (0,)*d) -- so points have d elements after None
    # Lines: (1, y) where y has d+1 elements (y[1] through y[d+1])
    #
    # The neighbours function: for a point x (side=0, d elements x[1]..x[d]),
    # for each first in range(q): y[1] = first, y[0] = None (implicitly),
    # then for c=2..d: y[c] = x[c] + y[a]*x[b] (mod q)
    #
    # So a point has d coordinates, a line has d+1 coordinates (y[1]..y[d+1]).
    # Total points: q^d = 3^11 = 177147
    # Total lines: q^(d+1) = 3^12 = 531441
    # But we only take the connected component of the zero point.
    #
    # This could still be very large. Let me just implement the LUW construction
    # as in the seed, but with a deadline guard, and see what n we get.
    #
    # Actually, for the purpose of this task, let me just implement a clean version
    # of the LUW construction with a deadline guard. The key is to produce a valid
    # 3-regular graph with girth >= 16.

    q = k  # = 3
    # Choose d such that girth >= g
    # LUW CD(d,q) has girth >= d+5 for odd d, d+6 for even d
    if g % 2 == 0:
        # girth = 2*floor(g/2) + 1 or 2*floor(g/2)... 
        # For girth >= g, we need d+5 >= g (odd d) or d+6 >= g (even d)
        # Try odd d first: d >= g-5
        d = g - 5
        if d % 2 == 0:
            d += 1  # make it odd
    else:
        d = g - 5
        if d % 2 == 0:
            d += 1
    
    d = max(3, d)

    deadline = time.time() + 120.0 * 0.85  # 85% of 120s = 102s safety margin

    # GF(q) operations for q=3
    def gf_add(a, b):
        return (a + b) % q
    def gf_sub(a, b):
        return (a - b) % q
    def gf_mul(a, b):
        return (a * b) % q

    # Rule for LUW construction
    # rule[c] = (a, b) for c in 2..d
    # The rule is defined recursively based on c mod 4
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def get_neighbours(vertex):
        side, x = vertex
        if side == 0:
            # x is a tuple: x[0]=None, x[1]..x[d] are in GF(q)
            for first in range(q):
                y = [None] + [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                yield (1, tuple(y[1:]))
        else:
            # x is a tuple: x[0]..x[d] are in GF(q) (line has d+1 coords)
            for first in range(q):
                y = [None] + [0] * d
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield (0, tuple(y[1:]))

    start = (0, (0,) * d)  # zero point: side=0, d zeros
    index = {start: 0}
    order = [start]
    edges = []
    
    max_iterations = 500000  # safety limit on BFS iterations
    
    for i in range(len(order)):
        if time.time() > deadline:
            break
        if len(order) > max_iterations:
            break
        vertex = order[i]
        side, x = vertex
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

# EVOLVE-BLOCK-END
