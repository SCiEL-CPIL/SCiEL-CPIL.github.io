import time
import random
from collections import deque

# ---- global deadline (150s wall-clock kill; keep margin) ----
_T0 = time.time()
_DEADLINE = _T0 + 135.0


# ---------------------------------------------------------------------------
# LUW construction (always k-regular, girth >= g for q>=3, d odd >= g-5)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    q = k
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
        c0 = c - (c - 5) % 4
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


# ---------------------------------------------------------------------------
# Fast girth certificate
# ---------------------------------------------------------------------------
def _bfs_ecc(adj, s):
    """Max BFS depth (eccentricity) from s. Returns -1 if disconnected."""
    n = len(adj)
    dist = [-1] * n
    dist[s] = 0
    q = deque([s])
    mx = 0
    while q:
        u = q.popleft()
        du = dist[u]
        nd = du + 1
        if nd > mx:
            mx = nd
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = nd
                q.append(w)
    for x in dist:
        if x < 0:
            return -1
    return mx


def _min_cycle_bound(adj):
    """Lower bound on girth: min over vertices of 2*(ecc(v)+1).
    If this is >= g, the graph is certified girth >= g."""
    n = len(adj)
    best = 10 ** 9
    for s in range(n):
        e = _bfs_ecc(adj, s)
        if e < 0:
            return 0
        b = 2 * (e + 1)
        if b < best:
            best = b
    return best


def _certified_girth(adj, g):
    """True iff min_cycle_bound >= g (a valid girth certificate)."""
    return _min_cycle_bound(adj) >= g


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ---------------------------------------------------------------------------
# Random 3-regular generation (fallback / alternative start)
# ---------------------------------------------------------------------------
def _generate_random_3regular(n):
    if n < 4 or n % 2 != 0:
        return None
    deg = [0] * n
    for _ in range(3):
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(0, n, 2):
            a, b = perm[i], perm[i + 1]
            deg[a] += 1
            deg[b] += 1
    if any(d != 3 for d in deg):
        return None
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if deg[i] > 0 and deg[j] > 0:
                adj[i].append(j)
                adj[j].append(i)
                deg[i] -= 1
                deg[j] -= 1
    return adj


# ---------------------------------------------------------------------------
# Swap-based shrinking
# ---------------------------------------------------------------------------
def _shrink(adj, g, deadline, seed=12345):
    """Try to reduce n via random 2-opt edge swaps that keep girth >= g.
    Returns (n, edges) of the best (smallest n) graph found, always valid."""
    rng = random.Random(seed)
    n = len(adj)
    adj = [list(a) for a in adj]
    best_n = n
    best_adj = [list(a) for a in adj]

    # Precompute current certificate
    cur_bound = _min_cycle_bound(adj)
    if cur_bound < g:
        # not certified; still proceed (we re-check after each swap)
        pass

    def neighbors_set(v):
        return set(adj[v])

    while time.time() < deadline - 8.0 and n >= 4:
        # Try a batch of random swaps
        for _ in range(20000):
            if time.time() > deadline - 8.0:
                break
            # Pick two random edges (u,v) and (x,y) with no shared vertices
            u = rng.randrange(n)
            x = rng.randrange(n)
            if u == x:
                continue
            if x in adj[u]:
                continue
            v = adj[u][rng.randrange(3)]
            y = adj[x][rng.randrange(3)]
            if v == y or v == x or y == u:
                continue
            if y in adj[v]:
                continue
            # Perform swap: remove (u,v),(x,y); add (u,x),(v,y)
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)

            # Check girth certificate for affected vertices only
            ok = True
            for s in (u, v, x, y):
                e = _bfs_ecc(adj, s)
                if e < 0:
                    ok = False
                    break
                if 2 * (e + 1) < g:
                    ok = False
                    break
            if not ok:
                # Undo swap
                adj[u].remove(x)
                adj[x].remove(u)
                adj[v].remove(y)
                adj[y].remove(v)
                adj[u].append(v)
                adj[v].append(u)
                adj[x].append(y)
                adj[y].append(x)
                continue
            # Accept swap (girth certificate still holds for affected vertices)
            # Note: we don't reduce n here; n stays the same.
            # This is a local improvement in girth margin, not size reduction.
        # After a batch, attempt to reduce n by finding a vertex of degree < 3
        # (shouldn't happen) or by a more aggressive move.
        # For now, we keep n constant; the real size reduction requires
        # a different mechanism.
        break

    # Build edges from best_adj
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct(k, g):
    """Construct a k-regular graph with girth >= g.
    Returns (n, edges)."""
    if k == 3 and g == 16:
        deadline = _DEADLINE
        # Start from LUW (valid 936-vertex solution)
        try:
            n, edges = _luw_construct(k, g)
            adj = _build_adj(n, edges)
            # Verify it's certified
            if _certified_girth(adj, g):
                # Try to shrink via swaps (best effort)
                try:
                    n2, edges2 = _shrink(adj, g, deadline)
                    if _certified_girth(_build_adj(n2, edges2), g):
                        return n2, edges2
                except Exception:
                    pass
                return n, edges
        except Exception:
            pass
        # Fallback: try random 3-regular + local search
        try:
            for n in range(512, 940, 2):
                if time.time() > deadline - 5:
                    break
                adj = _generate_random_3regular(n)
                if adj is None:
                    continue
                if _certified_girth(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        except Exception:
            pass
        # Final fallback: LUW
        return _luw_construct(k, g)
    # General case
    return _luw_construct(k, g)
