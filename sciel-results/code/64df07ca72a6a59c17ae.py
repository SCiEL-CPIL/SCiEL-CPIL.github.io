import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    if k == 4 and g == 10:
        r = _construct_4_10(start_time, deadline)
        if r is not None:
            return r
    return _luw_construct(k, g)

# ---------------------------------------------------------------------------
# (4,10)-cage via a Z_32 lift of K_{5,5} minus a perfect matching.
# Top fibers 0..4 (vertices 0..159), bottom fibers 0..4 (160..319).
# A lift is a bi-Cayley graph of G = C_32 : C_2 (order 64) with
#   a = (1,0)  (self-loop, order 32)
#   b = (0,1)  (matching, order 2),  b a b = a^{-1}
# so the "top" fiber i connects to bottom fiber j with voltage (a^{v[i][j]} b).
# ---------------------------------------------------------------------------
def _construct_4_10(start_time, deadline):
    n = 320
    gs = 32

    # ---- Strategy 1: exhaustive linear voltage family v[i][j] = (a*i + b*j) mod 32
    for a in range(gs):
        for b in range(gs):
            v = [[(a * i + b * j) % gs for j in range(5)] for i in range(5)]
            edges = _build_lift(v, gs)
            if _girth_ok(edges, n, 10):
                return n, edges

    # ---- Strategy 2: structured / heuristic families
    candidates = []
    # powers-of-2 patterns (kept for completeness)
    for base in (1, 2):
        v = [[0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    v[i][j] = (base ** ((j - i) % 5)) % gs
        candidates.append(v)
    # symmetric linear family (a, -a)
    for a in range(gs):
        v = [[(a * i - a * j) % gs for j in range(5)] for i in range(5)]
        candidates.append(v)
    # anti-symmetric family v[i][j] = -v[j][i]  (v[i][i]=0)
    for s in range(1, 11):
        v = [[0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(i + 1, 5):
                x = (s * (i + 1) * (j + 1)) % gs
                v[i][j] = x
                v[j][i] = (-x) % gs
        candidates.append(v)

    for v in candidates:
        edges = _build_lift(v, gs)
        if _girth_ok(edges, n, 10):
            return n, edges

    # ---- Strategy 3: time-bounded random search (fix v[0][1]=1 to break symmetry)
    random.seed(12345)
    while time.time() < deadline - 3.0:
        v = [[0] * 5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                if i == 0 and j == 1:
                    v[i][j] = 1
                else:
                    v[i][j] = random.randrange(gs)
        edges = _build_lift(v, gs)
        if _girth_ok(edges, n, 10):
            return n, edges

    return None  # fall back to LUW in construct()


def _build_lift(voltages, gs):
    edges = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            w = voltages[i][j]
            u0 = i * gs
            v0 = 5 * gs + j * gs
            for x in range(gs):
                edges.append((u0 + x, v0 + (x + w) % gs))
    return edges


def _girth_ok(edges, n, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Exact girth check via non-backtracking walks (no backtracking allowed).
    # A non-backtracking walk of length L with a repeated vertex gives a cycle
    # of length <= L.  So if no such walk of length min_girth-1 closes, girth >= min_girth.
    depth = min_girth - 1  # 9
    for start in range(n):
        # state: (vertex, forbidden_prev)
        cur = [(start, -1)]
        for _ in range(depth):
            nxt = []
            for u, prev in cur:
                for w in adj[u]:
                    if w == prev:
                        continue
                    if w == start:
                        return False
                    nxt.append((w, u))
            cur = nxt
            if not cur:
                break
    return True


def _luw_construct(k, g):
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                aa, bb = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[aa], x[bb]))
                else:
                    y[c] = sub(x[c], mul(x[aa], y[bb]))
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
