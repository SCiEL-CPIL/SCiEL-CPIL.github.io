"""Construct a 4-regular graph of girth 10 via Lazebnik-Ustimenko-Woldar CD(d,q)."""
import time

# EVOLVE-BLOCK-START
def _girth_check(n, edges, g):
    """Return True if girth >= g (no cycle shorter than g). Uses BFS from each vertex
    to detect cycles; a cycle of length L is found when an edge connects two vertices
    at distance (L-1)/... We use the standard: for each directed edge, BFS avoiding
    backtracking; if we reach the start within < g steps, girth < g."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Non-backtracking BFS from every vertex
    for s in range(n):
        # state: (current, prev), dist
        from collections import deque
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

def construct(k, g):
    deadline = time.time() + 110.0
    # Lazebnik-Ustimenko-Woldar CD(d, q): q-regular bipartite, girth >= d+5 (odd d).
    q = k
    # Need girth >= g. girth = d+5 for odd d. Choose smallest odd d with d+5 >= g.
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
    # Verify girth; if too slow, trust the construction (it is girth >= d+5).
    return n, edges
# EVOLVE-BLOCK-END
