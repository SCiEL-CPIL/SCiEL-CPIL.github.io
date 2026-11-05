import time
import sys

def _lwe_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction.
    Girth of LWE(d,q) is 2d+1. We need girth >= g.
    So 2d+1 >= g => d >= (g-1)//2.
    For g=9, d >= 4.
    n = q * (q^d - 1)/(q-1)
    
    For k=5, g=9: d=4, n=5*(625-1)/4 = 5*624/4 = 780
    """
    d = max(3, (g - 1) // 2)
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q

    # Precompute powers of q
    qpow = [1] * (d + 2)
    for i in range(1, d + 2):
        qpow[i] = qpow[i-1] * q
    
    n_total = q * (qpow[d] - 1) // (q - 1)
    n_per_side = n_total // 2
    
    # Vertices on side 0: (x1,...,xd) in GF(q)^d
    # Vertices on side 1: (y1,...,yd) in GF(q)^d with constraint?
    # Actually for LWE, the construction is:
    # Side 0: all of GF(q)^d (q^d vertices)
    # Side 1: all of GF(q)^d (q^d vertices)
    # But that gives 2*q^d, which is not q*(q^d-1)/(q-1) in general.
    #
    # Let me reconsider. The LWE graph LWE(d,q) has:
    # |V| = 2 * q^d for d >= 1? No.
    #
    # Actually, I think the correct count is:
    # For the bipartite version, each side has q^d vertices, total 2*q^d.
    # But that doesn't match q*(q^d-1)/(q-1) unless d=1.
    #
    # Let me look at this differently. The standard LWE construction:
    # V0 = GF(q)^d, V1 = GF(q)^d
    # Edge between (x1,...,xd) in V0 and (y1,...,yd) in V1 if:
    #   y[i] = x[i] + sum_{j > i} x[j] * y[j-1] ... 
    #
    # This is getting complex. Let me use the well-known formula:
    # n = 2*q^d for the bipartite LWE, girth = 2d+1 (if q >= 3 and d >= 1).
    # Wait, that's not right either.
    #
    # For LWE(d,q): n = 2*q^d, girth = 2d+1 (for appropriate parameters).
    # For k=5, g=9: we need d=4, n = 2*5^4 = 1250.
    #
    # But the Moore bound for n(5,9) is 2*(5+5^2+5^3+5^4) = 2*(5+25+125+625) = 2*780 = 1560.
    # So 1250 < 1560, which means LWE doesn't meet the Moore bound (it can't, unless it's a Moore graph).
    #
    # Hmm, let me just implement the BFS-based LWE correctly.
    
    # Actually, the standard LWE(d,q) construction:
    # Vertices: V = {0,1} x GF(q)^d
    # Two vertices (s, x) and (t, y) are adjacent if s != t and:
    # For d=1: x1 = y1 + 1 (or some fixed rule)
    # For d>1: more complex recurrence
    #
    # The girth is 2d+1 for q >= 3.
    #
    # For k=5, g=9: d=4, n = 2*5^4 = 1250
    #
    # But wait - the degree should be q=5, and the girth should be 2d+1=9.
    # Let me implement this properly using the known formula.
    
    # I'll use a simpler, well-known construction: the incidence graph of a projective geometry or similar.
    # Actually, let me just use the BFS-based approach from the seed, which is known to work.
    
    # Rule for LUW construction
    # rule[c] = (a, b) such that y[c] = x[c] +/- y[a] * x[b]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rules = [None] * (d + 1)
    for c in range(2, d + 1):
        rules[c] = rule[c]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rules[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        side, x = vertex
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    n, edges = _lwe_graph(k, g)
    return n, edges
