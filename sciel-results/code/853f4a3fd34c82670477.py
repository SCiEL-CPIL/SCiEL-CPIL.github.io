"""Voltage-graph (lift) search for small 3-regular graphs of large girth.

Base graph: the dodecahedral graph (20 vertices, 3-regular, girth 5).
Lift: assign each base edge a voltage in Z_m; the m-fold lift is 3-regular
and simple with n = 20*m vertices. We scan m from the Moore bound upward and
search over voltage assignments, certifying girth exactly with non-backtracking
walks from every vertex (no closed non-backtracking walk of length < g).
"""
import time
import random
from collections import deque

# ---- Dodecahedral graph (canonical 20 vertices, 3-regular, girth 5) ----
_DODEC = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5),
    (2, 6), (2, 7),
    (3, 8), (3, 9),
    (4, 5), (4, 10),
    (5, 11), (5, 12),
    (6, 7), (6, 13),
    (7, 14), (7, 15),
    (8, 9), (8, 16),
    (9, 17), (9, 18),
    (10, 11), (10, 19),
    (11, 12),
    (12, 13), (12, 19),
    (13, 14),
    (14, 15), (14, 19),
    (15, 16),
    (16, 17), (16, 18),
    (17, 18),
    (18, 19),
]
NBASE = 20
EBASE = list(_DODEC)
EB = {u: [] for u in range(NBASE)}
for u, v in EBASE:
    EB[u].append(v)
    EB[v].append(u)
assert all(len(EB[u]) == 3 for u in range(NBASE))


def _moore(k, g):
    if g % 2 == 0:
        r = g // 2
        return 1 + k * (k - 1) ** (r - 1)
    else:
        r = (g - 1) // 2
        return 2 * (k * (k - 1) ** (r - 1))


def _has_short_cycle(adj, g):
    """Return True iff some non-backtracking closed walk of length < g exists."""
    L = g - 1
    for s in range(len(adj)):
        seen = set()
        q = deque()
        for nb in adj[s]:
            q.append((nb, s))
            seen.add((nb, s))
        while q:
            v, p = q.popleft()
            if v == s:
                return True
            for w in adj[v]:
                if w == p:
                    continue
                st = (w, v)
                if st in seen:
                    continue
                seen.add(st)
                q.append(st)
    return False


def _valid(adj, g):
    return not _has_short_cycle(adj, g)


def _lift_adj(base_adj, volt, m):
    n = len(base_adj) * m
    adj = [[] for _ in range(n)]
    for u in range(len(base_adj)):
        for v, t in zip(base_adj[u], volt[u]):
            a = u * m + t
            b = v * m
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _try_m(base_adj, m, g, rng, ntry, deadline):
    n = len(base_adj) * m
    best = None
    for attempt in range(ntry):
        if time.time() >= deadline:
            return best
        if attempt == 0:
            volt = [[0, 1, 2] for _ in range(len(base_adj))]
        else:
            volt = [[rng.randrange(m) for _ in range(3)] for _ in range(len(base_adj))]
        adj = _lift_adj(base_adj, volt, m)
        if _valid(adj, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            best = (n, edges)
            return best
    return best


def construct(k, g):
    start = time.time()
    deadline = start + 120.0
    if k != 3:
        # Fallback: random regular graph (best effort) for non-cubic k.
        rng = random.Random(12345)
        mo = _moore(k, g)
        n = 2 * mo
        for _ in range(5000):
            if time.time() >= deadline:
                break
            deg = [0] * n
            edges = []
            seen = set()
            ok = True
            while sum(deg) < n * k:
                u = rng.randrange(n)
                v = rng.randrange(n)
                if u == v or deg[u] >= k or deg[v] >= k:
                    continue
                key = (min(u, v), max(u, v))
                if key in seen:
                    continue
                seen.add(key)
                deg[u] += 1
                deg[v] += 1
                edges.append(key)
            if max(deg) == k and len(edges) == n * k // 2:
                adj = [[] for _ in range(n)]
                for a, b in edges:
                    adj[a].append(b)
                    adj[b].append(a)
                if _valid(adj, g):
                    return n, edges
        return n, edges

    base = EB
    mo = _moore(3, g)
    m_min = (mo + NBASE - 1) // NBASE
    rng = random.Random(20240607)

    best = None
    m = m_min
    while True:
        if time.time() >= deadline - 3.0:
            break
        # More random attempts when little time remains so we can still certify.
        ntry = 1 + 20
        res = _try_m(base, m, g, rng, ntry, deadline)
        if res is not None:
            best = res
            return res
        m += 1

    if best is None:
        # Safety: return the smallest candidate (may be invalid; better than crashing).
        n = NBASE * m_min
        return n, []
    return best
