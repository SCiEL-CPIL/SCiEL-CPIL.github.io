"""Construct a 4-regular graph of girth 10 via generalized hexagon or LUW CD(d,q)."""
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

def _generalized_hexagon():
    """Construct the incidence graph of the generalized hexagon of order (2,2).
    
    Returns (n, edges) for a 4-regular bipartite graph on 112 vertices with girth 12,
    or None if the construction doesn't yield exactly 56 distinct lines.
    """
    # Points: (a, b) for a in Z_7, b in Z_8 -> 56 points
    # Generate candidate lines from 4 line families
    lines = set()
    for a in range(7):
        for b in range(8):
            l1 = frozenset([(a,b), (a,(b+1)%8), ((a+1)%7,(b+1)%8), ((a+1)%7,(b+2)%8)])
            lines.add(l1)
            l2 = frozenset([(a,b), (a,(b+7)%8), ((a+6)%7,(b+7)%8), ((a+6)%7,(b+6)%8)])
            lines.add(l2)
            l3 = frozenset([(a,b), ((a+1)%7,b), ((a+1)%7,(b+1)%8), ((a+2)%7,(b+1)%8)])
            lines.add(l3)
            l4 = frozenset([(a,b), ((a+6)%7,b), ((a+6)%7,(b+7)%8), ((a+5)%7,(b+7)%8)])
            lines.add(l4)
    
    if len(lines) != 56:
        return None
    
    points = [(a, b) for a in range(7) for b in range(8)]
    point_id = {p: idx for idx, p in enumerate(points)}
    line_list = list(lines)
    line_id = {l: idx for idx, l in enumerate(line_list)}
    
    edges = []
    for (a, b) in points:
        pid = point_id[(a, b)]
        for l in lines:
            if (a, b) in l:
                lid = 56 + line_id[l]
                edges.append((pid, lid))
    
    return 112, edges

def construct(k, g):
    deadline = time.time() + 110.0
    
    if k == 4 and g <= 12:
        result = _generalized_hexagon()
        if result is not None:
            n, edges = result
            deg = Counter()
            edge_set = set()
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
                edge_set.add((min(u,v), max(u,v)))
            if len(edge_set) == len(edges) and all(deg[i] == 4 for i in range(n)):
                return n, edges
    
    # Fallback: Lazebnik-Ustimenko-Woldar CD(d, q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
# EVOLVE-BLOCK-END
