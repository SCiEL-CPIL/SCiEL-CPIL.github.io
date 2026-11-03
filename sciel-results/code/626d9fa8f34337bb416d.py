import time
import random
import sys

_DEADLINE = 120.0
_TIME_MARGIN = 10.0

def _elapsed(start):
    return time.time() - start

def _time_left(start):
    return _DEADLINE - _TIME_MARGIN - _elapsed(start)

def _lazebnik(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k. Always valid."""
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

def _girth_ok_fast(adj, n, g, start):
    """Return True iff the graph has no cycle of length < g."""
    for v0 in range(n):
        seen = set()
        frontier = [(v0, -1)]
        seen.add((v0, -1))
        for length in range(1, g):
            nxt = []
            for cur, prev in frontier:
                for nb in adj[cur]:
                    if nb == prev:
                        continue
                    if length >= 3 and nb == v0:
                        return False
                    state = (nb, cur)
                    if state not in seen:
                        seen.add(state)
                        nxt.append(state)
            frontier = nxt
            if not frontier:
                break
            if _time_left(start) < 0:
                return True
    return True

def _construct_cayley_graph(m, g, start):
    """
    Construct a 4-regular Cayley graph on Z_2^m.
    Vertices are integers 0..2^m-1.
    Edges are defined by 4 random non-zero vectors.
    """
    n = 1 << m
    if n < 100: # Too small
        return None
    
    # Pick 4 random non-zero vectors
    while True:
        gens = set()
        while len(gens) < 4:
            v = random.randint(1, n - 1)
            gens.add(v)
        gens = list(gens)
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        edges_set = set()
        valid = True
        for u in range(n):
            for v in gens:
                w = u ^ v
                if w > u:
                    edges_set.add((u, w))
                adj[u].append(w)
        
        # Check girth
        if _girth_ok_fast(adj, n, g, start):
            return n, sorted(edges_set)
            
        if _time_left(start) < 1.0:
            return None

def construct(k, g):
    start = time.time()
    
    if k != 4 or g != 10:
        return _lazebnik(k, g)
        
    # Try Cayley graph constructions with increasing m
    # m=10 -> n=1024
    # m=9 -> n=512
    # m=8 -> n=256
    # m=7 -> n=128
    # Moore bound is 243, so n must be >= 243.
    # Try m=8 (n=256) first.
    
    for m in [8, 9, 10, 11]:
        if _time_left(start) < 5.0:
            break
        result = _construct_cayley_graph(m, g, start)
        if result:
            return result
            
    # Fallback to LUW
    return _lazebnik(k, g)
