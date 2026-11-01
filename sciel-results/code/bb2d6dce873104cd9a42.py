"""Construct k-regular graphs with girth at least g."""
import time
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill

    # For k=5, g=9: The known best is 1116 vertices.
    # The LUW construction gives a valid graph but with more vertices.
    
    # First, try the generalized quadrangle GQ(4,2) based construction.
    # This should give a 5-regular graph with girth 9 (or better) on fewer vertices.
    
    if k == 5 and g <= 9:
        try:
            n, edges = _gq42_construct()
            if n < 1116:  # Only use if it's better than the known cage
                return n, edges
        except Exception:
            pass
    
    # Fallback: LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

def _luw_construct(k, g):
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

def _gq42_construct():
    """
    Construct a 5-regular graph with girth 9 using the generalized quadrangle GQ(4,2).
    
    The GQ(4,2) is a generalized quadrangle with parameters (s,t) = (4,2).
    It has (s+1)(st+1) = 5*9 = 45 points and 45 lines.
    Each point is on t+1 = 3 lines, each line has s+1 = 5 points.
    
    The incidence graph of GQ(4,2) is a 3-regular bipartite graph on 90 vertices
    with girth 8 (since generalized quadrangles have no cycles of length < 4*2 = 8,
    actually girth is 8 for GQ(s,t) when s,t >= 2).
    
    To get a 5-regular graph with girth 9, we can use a different approach.
    
    Actually, let's use the known construction for the (5,9)-cage.
    The smallest known (5,9)-cage has 1116 vertices.
    
    Instead, let's try to construct a smaller graph using a different method.
    
    For now, let's return a valid 5-regular graph with girth >= 9.
    We can use the LUW construction with d=5, which gives girth >= 10.
    But that gives n=6250, which is worse than 1116.
    
    Let's try a different approach: use a random regular graph and check girth.
    For n around 500-1000, this might be feasible.
    
    Actually, let's just return the LUW construction result for now, since it's guaranteed valid.
    """
    # For k=5, g=9, the LUW construction with d=5 gives a valid graph.
    # But n=6250 is too large.
    
    # Let's try to construct a smaller graph.
    # The (5,9)-cage is known to have at least 428 vertices (lower bound).
    # The best known upper bound is 1116.
    
    # Without a specific construction, let's try to generate a random 5-regular
    # graph on n vertices and check if it has girth >= 9.
    # We'll try n from 428 upward.
    
    # This is computationally expensive, so let's be careful.
    # For now, let's just return a valid (but not optimal) graph.
    
    # Actually, let's use a different algebraic construction.
    # The "Tutte" construction or "Bose-Chowla" might work.
    
    # For simplicity, let's return the LUW construction result.
    # This is valid but not optimal.
    
    # Let's try a smaller construction using finite fields.
    # For k=5, g=9, we can try to use a Cayley graph over Z_n or similar.
    
    # Actually, let's just raise an exception to fall back to LUW.
    raise ValueError("GQ construction not implemented")
