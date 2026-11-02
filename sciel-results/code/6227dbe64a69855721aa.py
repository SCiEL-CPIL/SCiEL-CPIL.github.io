import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Try Cayley graph construction on GF(q)^3
    # We want a k-regular graph.
    # For q=11, n=1331.
    # Generators: 5 elements in GF(11)^3.
    # We need to choose generators such that the girth is at least g.
    
    def try_cayley(q, k, g):
        n = q**3
        # Vertices are 3-tuples over GF(q)
        # We need k generators.
        # Since the graph must be undirected, generators come in pairs +/- v.
        # If k is odd, we can't have a symmetric set of odd size unless one is 0 (loop, not allowed) or we use a different structure.
        # Wait, Cayley graphs of abelian groups are bipartite if all generators have odd order?
        # Actually, for undirected Cayley graph, S = S^{-1}.
        # If k=5, we need 5 generators. This implies one generator is self-inverse?
        # In GF(q)^3, v = -v implies 2v=0. If q is odd, v=0. Loop.
        # So for odd k and odd q, we cannot form a simple undirected Cayley graph using only group elements?
        # Unless we allow directed? No, graph is simple undirected.
        # So Cayley graphs of abelian groups of odd order have even degree.
        # So k=5 is impossible for GF(q)^3 with odd q.
        
        # What if q is even? q=8? n=512.
        # GF(8) is not a field in the standard sense for Python integers, but it's a vector space over GF(2).
        # We can use GF(2)^3? n=8. Too small.
        # GF(2)^9? n=512.
        # Degree 5 is possible in GF(2)^m.
        # Let's try q=2, dim=9. n=512.
        # Generators: 5 vectors in GF(2)^9.
        # S = {v1, ..., v5}. Since -v = v in GF(2), S is symmetric.
        # We need girth >= 9.
        
        pass

    # LZW Construction
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
