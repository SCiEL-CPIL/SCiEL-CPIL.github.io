import time
from itertools import combinations

# ---- Precompute edge universe (120 edges on 16 vertices) ----
N = 16
ALL = list(combinations(range(N), 2))
M = len(ALL)  # 120
E2I = {e: i for i, e in enumerate(ALL)}
E2I.update({(b, a): i for (a, b), i in E2I.items()})

# For each edge i=(u,v): list of (w, i_uw, i_vw) whose presence together with i forms a triangle
WIT = [[] for _ in range(M)]
for i in range(M):
    u, v = ALL[i]
    for w in range(N):
        if w == u or w == v:
            continue
        WIT[i].append((w, E2I[(u, w)], E2I[(v, w)]))

# For each vertex v: list of edge indices incident to v
INC = [[] for _ in range(N)]
for i in range(M):
    u, v = ALL[i]
    INC[u].append(i)
    INC[v].append(i)

# ---- Clebsch graph (40 edges), 16 vertices, triangle-free, 4-chromatic ----
def _clebsch_edges():
    e = []
    for i in range(16):
        for d in (1, 2, 5, 10):
            j = (i + d) % 16
            if i < j:
                e.append((i, j))
    return e

CLEBSCH = _clebsch_edges()

# ---- Fast bitset 3-colorability (returns True if 3-colorable) ----
def _is_3colorable(adj):
    # adj[v] = bitmask of neighbors of v
    n = N
    order = sorted(range(n), key=lambda v: -adj[v].bit_count())
    color = [-1] * n

    def bt(pos):
        if pos == n:
            return True
        v = order[pos]
        nb = adj[v]
        while nb:
            lo = nb & -nb
            u = lo.bit_length() - 1
            nb ^= lo
            if color[u] == -1:
                return True  # uncolored neighbor => order violated (shouldn't happen)
        used = 0
        nb = adj[v]
        while nb:
            lo = nb & -nb
            u = lo.bit_length() - 1
            nb ^= lo
            c = color[u]
            if c >= 0:
                used |= 1 << c
        for c in range(3):
            if not (used >> c) & 1:
                color[v] = c
                if bt(pos + 1):
                    return True
                color[v] = -1
        return False

    return bt(0)

# ---- Fast triangle-free check + edge count from an edge list ----
def _validate(edges):
    adj = [0] * N
    for (u, v) in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    for v in range(N):
        nb = adj[v]
        while nb:
            lo = nb & -nb
            u = lo.bit_length() - 1
            nb ^= lo
            if adj[u] & (1 << v) and (adj[u] & adj[v]):
                return False  # common neighbor => triangle
    return True

def _is_valid(edges):
    if not _validate(edges):
        return False
    adj = [0] * N
    for (u, v) in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return not _is_3colorable(adj)

# ---- Build initial state from an edge list ----
def _build_state(edges):
    x = [0] * M
    adj = [0] * N
    for (u, v) in edges:
        i = E2I[(u, v)]
        x[i] = 1
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return x, adj

def _edges_from_x(x):
    return [list(ALL[i]) for i in range(M) if x[i]]

# ---- Local search (WalkSAT/Max-SAT style): edge-presence soft, triangle-free hard ----
def _local_search(x0, adj0, deadline, rng):
    x = x0[:]
    adj = adj0[:]
    best_x = x[:]
    best_n = sum(x)

    def _check(i):
        # return True if flipping edge i keeps graph triangle-free
        u, v = ALL[i]
        common = adj[u] & adj[v]
        if x[i]:
            # removing edge: check no other common neighbor remains
            if common & ~(1 << u) & ~(1 << v):
                return False
            return True
        else:
            # adding edge: need no common neighbor
            if common & ~(1 << u) & ~(1 << v):
                return False
            return True

    def _flip(i):
        u, v = ALL[i]
        if x[i]:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            x[i] = 0
        else:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            x[i] = 1

    # 3-colorability check (only on triangle-free states)
    def _ok():
        return not _is_3colorable(adj)

    if not _ok():
        return best_x, best_n

    while time.time() < deadline:
        # random edge flip attempt
        i = rng.randrange(M)
        if not _check(i):
            continue
        _flip(i)
        if _ok():
            cur = sum(x)
            if cur > best_n:
                best_n = cur
                best_x = x[:]
        else:
            # not 3-colorable -> undo
            _flip(i)
    return best_x, best_n

def construct():
    deadline = time.time() + 110  # safety margin before 150s wall-clock kill

    # Start from the known optimal Clebsch graph (40 edges)
    base_x, base_adj = _build_state(CLEBSCH)
    assert _validate(CLEBSCH)
    assert not _is_3colorable(base_adj), "Clebsch must be 4-chromatic"

    import random
    rng = random.Random(12345)

    best_x = base_x[:]
    best_n = sum(base_x)  # 40

    # Local search to try to exceed 40 (should not, but be safe)
    if time.time() < deadline - 5:
        lx, ln = _local_search(base_x, base_adj, deadline - 5, rng)
        if ln > best_n:
            best_n = ln
            best_x = lx

    # A few random restarts to be robust (cheap, bounded)
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            # random maximal triangle-free graph
            x = [0] * M
            adj = [0] * N
            perm = list(range(M))
            rng.shuffle(perm)
            for i in perm:
                u, v = ALL[i]
                if (adj[u] & adj[v]) & ~(1 << u) & ~(1 << v):
                    continue
                x[i] = 1
                adj[u] |= 1 << v
                adj[v] |= 1 << u
            if _is_3colorable(adj):
                continue
            n = sum(x)
            if n > best_n:
                best_n = n
                best_x = x[:]

    return _edges_from_x(best_x)
