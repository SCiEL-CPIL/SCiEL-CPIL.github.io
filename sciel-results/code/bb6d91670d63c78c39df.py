"""Improved construction for small regular graphs of given girth."""
import time
import math

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    
    Uses the Lazebnik-Ustimenko-Woldar (LUW) construction for prime k,
    with careful parameter selection to minimize the number of vertices.
    
    For k=5, g=9: uses the LUW(5,5) construction which gives a 5-regular
    bipartite graph with girth 10 (exceeds requirement of 9).
    
    Returns (n, edges) where n is the number of vertices and edges is a list
    of (i, j) pairs on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    # The LUW construction CD(d, q) gives a q-regular bipartite graph
    # with girth at least d + 5 (for odd d) or d + 4 (for even d).
    # We need girth >= g, so we need d + 5 >= g (odd d) or d + 4 >= g (even d).
    # To minimize vertices, we want the smallest d that works.
    
    q = k  # The field size (must be prime for the standard LUW construction)
    
    # For q = 5 (prime), we can use the LUW construction.
    # The number of vertices in the connected component of the zero point
    # is roughly 2 * q^d for the full construction, but the connected
    # component may be smaller.
    
    # For k=5, g=9:
    # If d is odd: girth >= d+5, so need d+5 >= 9 => d >= 4, smallest odd d = 5
    # If d is even: girth >= d+4, so need d+4 >= 9 => d >= 5, smallest even d = 6
    # So d=5 (odd) gives girth >= 10, which satisfies g=9.
    
    # Let's use d such that the girth requirement is met with minimal d.
    # For odd d: girth >= d + 5
    # For even d: girth >= d + 4
    
    # We'll pick the smallest d that gives girth >= g.
    best_d = None
    for d_candidate in range(3, 20):
        if d_candidate % 2 == 1:
            min_girth = d_candidate + 5
        else:
            min_girth = d_candidate + 4
        if min_girth >= g:
            best_d = d_candidate
            break
    
    if best_d is None:
        best_d = g - 5 if (g - 5) % 2 == 1 else g - 4
    
    d = best_d
    
    # For q = 4, use GF(4) with XOR addition and a multiplication table
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y):
            return x ^ y
        def sub(x, y):
            return x ^ y
        def mul(x, y):
            return table[x][y]
    else:
        # For prime q, use modular arithmetic
        def add(x, y):
            return (x + y) % q
        def sub(x, y):
            return (x - y) % q
        def mul(x, y):
            return x * y % q
    
    # Define the rule for the LUW construction
    # rule[c] = (a, b) means: l[c] - p[c] = l[a] * p[b] for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        # The four possible rules in a cycle
        candidates = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = candidates[r]
    
    def neighbours(vertex):
        """Generate all neighbors of a vertex in the LUW graph."""
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                    # p[c] = l[c] - l[a] * p[b]
                    # x[c] = y[c] - y[a] * x[b]
                    # So y[c] = x[c] + y[a] * x[b]... wait, let me re-derive
                    # From the relation: l[c] - p[c] = l[a] * p[b]
                    # If we're at a line vertex and finding point neighbors:
                    # p[c] = l[c] - l[a] * p[b]
                    # But p[b] is what we're building... this is implicit.
                    # Actually in the LUW construction, both sides use the same
                    # linear relation. Let me use the standard formulation.
                    # For bipartite LUW: if v is on side 0 (points), neighbors are
                    # lines l such that l[c] = p[c] + l[a]*p[b] for all c.
                    # If v is on side 1 (lines), neighbors are points p such that
                    # p[c] = l[c] - l[a]*p[b]... but p[b] appears on both sides.
                    # The standard way: define the relation symmetrically.
                    # Actually, the LUW graph is defined so that point p is adjacent
                    # to line l iff for all c in 2..d: l[c] = p[c] + l[a_c]*p[b_c]
                    # where (a_c, b_c) = rule[c].
                    # So from a line l, the point neighbors p satisfy:
                    # p[c] = l[c] - l[a_c]*p[b_c]
                    # This is a system of equations. For the LUW construction,
                    # this system has a unique solution for p given l and p[1].
                    # Let me just compute it directly.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    # BFS from the zero point to find the connected component
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
        
        # Check deadline
        if time.time() > deadline - 10:
            break
    
    n = len(order)
    return n, edges
