"""Construct a 5-regular graph of girth 9 via line graph of a 3-regular girth-14 graph."""
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: take the line graph of a 3-regular graph with girth 14.
    
    Note: The standard line graph L(G) of a 3-regular graph is 4-regular.
    To get a 5-regular graph, we use the "extended line graph" or add a perfect matching.
    
    Actually, for the line graph approach to yield a 5-regular graph from a 3-regular
    base graph, we can use the following construction:
    - Start with a 3-regular graph G of girth >= 14
    - The line graph L(G) is 4-regular on 3n/2 vertices
    - Add a perfect matching to make it 5-regular
    
    However, adding a perfect matching may introduce short cycles.
    
    Alternative interpretation: Use a different construction that yields 5-regular.
    Let's try: if G is 3-regular, then L(G) has degree 4. We need degree 5.
    
    Let me instead construct a 3-regular graph of girth >= 14 using the LUW method
    or other methods, then take its line graph and augment it.
    
    For now, let me implement the LUW construction for 3-regular girth-14 graphs,
    then take the line graph.
    """
    if k != 5 or g != 9:
        # Fallback: use the original LUW construction for other parameters
        return _luw_construct(k, g)
    
    # For k=5, g=9: 
    # Step 1: Construct a 3-regular graph with girth >= 14
    # Step 2: Take its line graph (which is 4-regular)
    # Step 3: Add edges to make it 5-regular while maintaining girth >= 9
    
    # First, let's construct a 3-regular graph of girth >= 14
    # Using the LUW construction with q=3, d such that girth >= 14
    # LUW CD(d, q) has girth at least d+5 for odd d
    # So we need d+5 >= 14, i.e., d >= 9
    
    # For q=3, d=9: n = 2 * 3^9 = 39366 vertices for the 3-regular graph
    # Line graph would have 3 * 39366 / 2 = 59049 vertices - too large
    
    # Let me try a smaller construction. 
    # Actually, the LUW construction gives n = 2*q^d vertices.
    # For q=3, d=9: 2*3^9 = 39366. Too large.
    
    # Let me try a different approach: use a known small 3-regular graph of high girth
    # or use a random construction.
    
    # Construct a 3-regular graph of girth >= 14 using LUW
    # Then take its line graph and add a perfect matching to make it 5-regular
    
    # LUW CD(d, q) for q=3: n = 2 * 3^d vertices
    # Girth >= d + 5
    # For girth >= 14: d >= 9, but that gives 2*3^9 = 39366 vertices
    # Line graph would have 3 * 39366 / 2 = 59049 vertices - too large
    
    # Let me try a smaller d and check if girth is sufficient
    # For d=7: girth >= 12, n = 2*3^7 = 4374, line graph = 6561 vertices
    # For d=8: girth >= 13, n = 2*3^8 = 13122, line graph = 19683 vertices
    
    # Let me try d=7 first (girth 12 for base, need to verify line graph girth)
    # The line graph of a graph with girth g has girth related to g
    # A cycle of length c in G becomes a cycle of length c in L(G) if c >= 3
    # But also, if G has a vertex of degree 3, the three edges incident to it
    # form a triangle in L(G)! So L(G) always has triangles if G has vertices of degree >= 3.
    
    # The line graph approach doesn't work because L(G) of a 3-regular G has triangles.
    # Let me use the LUW construction directly for 5-regular girth-9 graphs.
    
    # LUW CD(d, q) with q=5: n = 2 * 5^d vertices
    # Girth >= d + 5 (for odd d)
    # For girth >= 9: d >= 4, but d must be odd, so d >= 5
    # For d=5: n = 2 * 5^5 = 6250, girth >= 10
    # For d=7: n = 2 * 5^7 = 156250, girth >= 12
    
    # Let me use d=5, q=5: n = 6250, girth >= 10 >= 9
    # This is much better than the current d=4 (n=2500, girth>=9) or d=5
    return _luw_construct(5, 9)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction - optimized."""
    q = k
    # For girth g, we need d such that the LUW graph has girth at least g.
    # The LUW graph CD(d, q) has girth at least d + 5 for odd d.
    # So we need d + 5 >= g, i.e., d >= g - 5.
    # For odd d: d = g - 5 if (g - 5) is odd, else g - 4.
    
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for each coordinate
    rule = {}
    # For c = 2, 3, 4, use specific rules
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    # Precompute multiplication tables for speed
    if q != 4:
        mul_table = [[(x * y) % q for y in range(q)] for x in range(q)]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    if q == 4:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = (x[c] + mul_table[y[a]][x[b]]) % q
                else:
                    if q == 4:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                    else:
                        y[c] = (x[c] - mul_table[x[a]][y[b]]) % q
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
