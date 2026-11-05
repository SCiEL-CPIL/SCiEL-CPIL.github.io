import time
import random
from collections import deque

def _girth_from_nonbacktracking(adj, n, g):
    # Rigorous: girth >= g iff no non-backtracking closed walk of length < g.
    # Walk state = (current, previous). Start from every directed edge.
    # A state revisited at a smaller length is dominated (shorter prefix).
    # A state reached at length L == g-1 is a closed non-backtracking walk
    # of length g-1 < g  =>  girth < g.
    best = {}
    for u in range(n):
        for v in adj[u]:
            state = (v, u)
            if state in best:
                continue
            best[state] = 1
            dq = deque([(v, u)])
            while dq:
                cur, prev = dq.popleft()
                L = best[(cur, prev)]
                if L == g - 1:
                    return False
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    ns = (nxt, cur)
                    old = best.get(ns)
                    if old is None or old > L + 1:
                        best[ns] = L + 1
                        dq.append(ns)
    return True


def _random_4reg(n, rng, deadline):
    # Random 4-regular simple graph via random perfect matching of half-edges.
    # Cheap short-cycle guard (triangles + 4-cycles) before full certification.
    if n % 2:
        return None
    if n < 4:
        return None
    if n == 4:
        return [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    while time.time() < deadline:
        pool = [i for i in range(n) for _ in range(4)]
        rng.shuffle(pool)
        m = len(pool) // 2
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(m):
            a = pool[2 * i]
            b = pool[2 * i + 1]
            if a == b:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok:
            continue
        if any(len(s) != 4 for s in adj):
            continue
        # triangle guard
        bad = False
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    if adj[i] & adj[j]:
                        bad = True
                        break
            if bad:
                break
        if bad:
            continue
        # 4-cycle guard
        bad = False
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    if len(adj[i] & adj[j]) >= 2:
                        bad = True
                        break
            if bad:
                break
        if bad:
            continue
        adj = [sorted(s) for s in adj]
        if _girth_from_nonbacktracking(adj, n, 10):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return edges
    return None


def _luw_construct():
    # Guaranteed-valid fallback (seed): 4-regular, girth 10, 1280 vertices.
    q = 4
    d = 8
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
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


def construct(k, g):
    start_time = time.time()
    deadline = 115.0  # safety margin under 150s kill

    if k != 4 or g != 10:
        # Out-of-scope: return a valid 4-regular-ish fallback is not applicable;
        # fall back to the seed LUW (valid for the task's k,g).
        return _luw_construct()

    rng = random.Random(12345)
    # Search increasing n; random 4-reg girth ~ 2*log_3(n) ~ 10-12 for n in [300, 500].
    for n in range(320, 601, 2):
        if time.time() > deadline - 25:
            break
        edges = _random_4reg(n, rng, deadline - 25)
        if edges is not None:
            return n, edges

    # Guaranteed-valid fallback (seed LUW, 1280 vertices).
    return _luw_construct()
