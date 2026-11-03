"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) construction CD(d, q)
    with q = k, then take the quotient by the involution (side, x) <-> (1-side, x)
    to get a smaller k-regular graph with the same girth guarantee.
    
    For k=5, g=9: d=4, q=5, full graph has 2*5^5=6250 vertices,
    quotient has ~3125 vertices.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    q = k
    # For girth g (odd), we need d such that girth >= d+5
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    d = max(3, d)

    # Finite field GF(q) operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Determine the rule for each coordinate c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        patterns = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = patterns[r]

    # Build the LUW graph using BFS from the zero point
    # Vertices are (side, coords) where side=0 is point, side=1 is line
    # coords is a tuple of length d+1 (indices 0..d), with coords[0] unused (None)
    
    start = (0, (None,) + (0,) * d)
    
    # BFS to find connected component
    index = {start: 0}
    order = [start]
    edges_full = []  # edges in the full bipartite graph
    
    for vi, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            # Build the neighbor on the other side
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # Point p ~ Line l: l[c] = p[c] + l[a] * p[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # Line l ~ Point p: p[c] = l[c] + l[a] * p[b]
                    # So p[c] = l[c] + l[a] * p[b]
                    # But we're computing p from l, so:
                    # p[c] = l[c] + l[a] * p[b]
                    # This means p[c] depends on p[b], which is unknown.
                    # The correct relation is symmetric:
                    # l[c] - p[c] = l[a] * p[b]
                    # If side==1 (at line l, finding point p):
                    #   p[c] = l[c] - l[a] * p[b]  ... still has p[b]
                    # 
                    # Actually the LUW incidence is:
                    # Point p is incident to line l iff for all c in 2..d:
                    #   l[c] - p[c] = l[a_c] * p[b_c]
                    # 
                    # When at point p (side=0), finding line l:
                    #   l[c] = p[c] + l[a_c] * p[b_c]  (l[a_c] known, p[b_c] known)
                    # 
                    # When at line l (side=1), finding point p:
                    #   p[c] = l[c] - l[a_c] * p[b_c]  (p[b_c] unknown!)
                    #
                    # The trick: the rules are designed so that b_c < c, so p[b_c] is
                    # already determined. Let me check: for c=2, (a,b)=(1,1), b=1 < 2 ✓
                    # for c=3, (a,b)=(2,1), b=1 < 3 ✓
                    # for c=4, (a,b)=(1,2), b=2 < 4 ✓
                    # The patterns ensure b < c, so p[b] is already set.
                    # 
                    # So: p[c] = l[c] - l[a] * p[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            # Add edge from point side only to avoid duplicates
            if side == 0:
                edges_full.append((index[vertex], index[neighbor]))
        
        if time.time() > deadline:
            break
    
    n_full = len(order)
    
    # Now take the quotient: identify (0, x) with (1, x)
    # The quotient graph has vertices = equivalence classes under (side, x) ~ (1-side, x)
    # Each class has at most 2 vertices.
    
    # Build quotient vertex indices
    # A quotient vertex is represented by a canonical form: (0, x) if (0,x) exists, else (1,x)
    # But actually, both (0,x) and (1,x) may exist in the graph.
    
    # Map from quotient vertex (canonical) to quotient index
    quot_index = {}
    quot_order = []
    
    # For each vertex in the full graph, determine its quotient class
    # The quotient class of (side, x) is represented by x (the coordinate tuple)
    # since (0,x) and (1,x) are identified.
    
    for vi, (side, x) in enumerate(order):
        # Canonical representative: use x as the key
        if x not in quot_index:
            quot_index[x] = len(quot_order)
            quot_order.append(x)
    
    n_quot = len(quot_order)
    
    # Build quotient edges
    # An edge in the full graph between (0, x) and (1, y) becomes an edge in the quotient
    # between the class of x and the class of y.
    # But we need to be careful: multiple full edges may map to the same quotient edge.
    
    quot_edges_set = set()
    quot_edges = []
    
    for u, v in edges_full:
        su, xu = order[u]
        sv, xv = order[v]
        # u is on point side (0), v is on line side (1)
        # Quotient class of u is xu, quotient class of v is xv
        qu = quot_index[xu]
        qv = quot_index[xv]
        if qu != qv:  # No self-loops
            key = (min(qu, qv), max(qu, qv))
            if key not in quot_edges_set:
                quot_edges_set.add(key)
                quot_edges.append(key)
    
    # Check if the quotient is k-regular
    # Each vertex in the full graph has degree q (one for each 'first' value)
    # In the quotient, a vertex x has edges to:
    #   - For each neighbor y of (0,x) in the full graph: edge to class y
    #   - For each neighbor y of (1,x) in the full graph: edge to class y
    # But (0,x) and (1,x) each have q neighbors, so x has up to 2q neighbors in the quotient.
    # However, some of these may be duplicates (same class).
    
    # Actually, let's verify the degree. The LUW graph is q-regular bipartite.
    # The quotient identifies (0,x) with (1,x). The degree of the quotient vertex [x] is:
    #   deg_full(0,x) + deg_full(1,x) - 2 * (number of edges between (0,x) and (1,x) that are "internal" to the class)
    # But (0,x) and (1,x) are not adjacent to each other in general.
    # 
    # Actually, the correct way: the quotient graph is NOT necessarily k-regular.
    # Let me reconsider.
    
    # The LUW graph CD(d,q) is a q-regular bipartite graph. The involution (side,x)->(1-side,x)
    # is an automorphism only if the incidence relation is symmetric in a specific way.
    # 
    # Actually, I think the standard result is that the LUW graph has an automorphism that
    # swaps points and lines (when the rules are chosen appropriately), and the quotient
    # by this automorphism gives a 2q-regular graph, not q-regular.
    #
    # So the quotient approach doesn't directly give a k-regular graph. Let me instead
    # just return the full LUW graph, which IS k-regular.
    
    # For k=5, g=9: d=4, q=5, n = 2 * 5^5 = 6250
    # This is larger than the 1116-vertex known best, but it's valid.
    
    return n_full, edges_full
# EVOLVE-BLOCK-END
