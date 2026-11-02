"""Construct a 4-regular graph of girth 10 via algebraic constructions."""
import time
from collections import deque, Counter

# EVOLVE-BLOCK-START
def _girth_check(n, edges, g):
    """Return True if girth >= g (no cycle shorter than g)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for s in range(n):
        dist = {}
        dq = deque()
        for nb in adj[s]:
            dq.append((nb, s, 1))
            dist[(nb, s)] = 1
        while dq:
            cur, prev, d = dq.popleft()
            if cur == s and d < g:
                return False
            if d >= g - 1:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if (nb, cur) not in dist:
                    dist[(nb, cur)] = d + 1
                    dq.append((nb, cur, d + 1))
    return True

def _validate(n, edges, k, g):
    """Validate: simple graph, degree k, girth >= g."""
    deg = Counter()
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        deg[u] += 1
        deg[v] += 1
        edge_set.add((min(u, v), max(u, v)))
    if len(edge_set) != len(edges):
        return False
    for i in range(n):
        if deg[i] != k:
            return False
    return _girth_check(n, edges, g)

def _luw_cd(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    d = max(3, d)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x ^ y
        sub = lambda x,y: x ^ y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q
    
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5) % 4, c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    n = len(order)
    return n, edges

def _circulant_construction(k, g):
    """Try a circulant or algebraic construction for small (k,g)."""
    if k != 4 or g != 10:
        return None
    # Use the known (4,10) cage: 320 vertices from Exoo et al.
    # We use a voltage graph / lifting approach on a 4-regular base.
    # For now, return None to fall through to LUW.
    return None

def construct(k, g):
    deadline = time.time() + 100.0
    
    # Try circulant/algebraic first (fast if it works)
    result = _circulant_construction(k, g)
    if result is not None:
        n, edges = result
        if _validate(n, edges, k, g):
            return n, edges
    
    # LUW CD(d,q) construction
    n, edges = _luw_cd(k, g, deadline)
    if _validate(n, edges, k, g):
        return n, edges
    
    # Last resort: return whatever we have (may be incomplete)
    return n, edges
# EVOLVE-BLOCK-END
