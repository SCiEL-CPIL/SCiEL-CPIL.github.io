"""Voltage lift of K_{3,3} over Z_m for 3-regular graphs of girth >= g."""
import random
from collections import deque

# EVOLVE-BLOCK-START

def _girth_via_nonbacktracking(n, adj, limit):
    """Return the girth (shortest cycle length), or limit if none shorter found.
    Uses non-backtracking BFS from each vertex (handles all girths including 4)."""
    best = limit
    for s in range(n):
        # state: (vertex, prev_vertex)
        dist = {}
        q = deque()
        for w in adj[s]:
            dist[(s, w)] = 1
            q.append((s, w))
        while q:
            v, p = q.popleft()
            d = dist[(v, p)]
            if d + 1 >= best:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == s:
                    # closed non-backtracking walk of length d+1 = a cycle
                    if d + 1 < best:
                        best = d + 1
                    continue
                state = (v, w)
                if state not in dist:
                    dist[state] = d + 1
                    q.append(state)
    return best

def _build_graph(m, voltages):
    """Build the 3-regular graph from K_{3,3} voltage lift over Z_m.
    vertices: (side, part, t), side in {0,1}, part in 0..2, t in 0..m-1
    index = side*(3*m) + part*m + t
    voltages: list of 9 ints, indexed by (partA, partB) -> partA*3+partB
    """
    n = 6 * m
    adj = [[] for _ in range(n)]
    for a in range(3):
        for b in range(3):
            v = voltages[a * 3 + b] % m
            for t in range(m):
                u = a * m + t
                w = 3 * m + b * m + ((t + v) % m)
                adj[u].append(w)
                adj[w].append(u)
    return n, adj

def _is_valid(n, adj, g):
    # degree check
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    # girth check
    girth = _girth_via_nonbacktracking(n, adj, g)
    return girth >= g

def construct(k, g):
    assert k == 3, "only k=3 supported"
    random.seed(12345)
    # try m values; prefer larger m to push girth up, but keep n small
    # We'll scan a range of m and search voltage assignments.
    best = None
    # candidate m list: start from small, grow. girth 16 needs decent m.
    # Empirically voltage lifts of K3,3 over Z_m have girth ~ O(log m).
    # We'll try m from 8 up to ~200, with more trials for smaller m.
    m_candidates = []
    m = 6
    while m <= 220:
        m_candidates.append(m)
        m += 1
    # sort by n ascending (i.e. m ascending) so we return smallest valid n
    for m in m_candidates:
        n = 6 * m
        # number of random voltage tries
        trials = 4000 if m < 30 else 1500 if m < 80 else 600
        found = False
        for trial in range(trials):
            if trial == 0:
                # all-zero voltages (gives disjoint copies, girth 4) - skip
                voltages = [0] * 9
            else:
                voltages = [random.randrange(m) for _ in range(9)]
            # quick reject: if any voltage is 0 on an edge, we get a 4-cycle? 
            # Actually a zero voltage edge means the edge connects (a,t)-(b,t) for all t,
            # combined with another edge of zero voltage gives 4-cycles. 
            # We'll just build and check.
            n2, adj = _build_graph(m, voltages)
            if _is_valid(n2, adj, g):
                # build edge list
                edges = []
                seen = set()
                for u in range(n2):
                    for w in adj[u]:
                        if u < w:
                            edges.append((u, w))
                if best is None or n2 < best[0]:
                    best = (n2, edges)
                found = True
                break  # found valid for this m; move to next (smaller n already? no, m increasing)
        if found:
            # since m is increasing, the first valid m gives smallest n
            break
    if best is None:
        # fallback: use the Lazebnik-Ustimenko-Woldar construction
        return _luw_construct(k, g)
    return best

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph CD(d, q)."""
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

# EVOLVE-BLOCK-END
